"""Structural validation of the Kubernetes manifests.

`kubectl --dry-run=client` still needs a reachable cluster for API discovery, so
CI cannot use it. These checks catch the failure modes that actually bite in
practice — a PVC name that does not match a claim, a probe pointing at the wrong
port, a workload with no namespace — without needing a cluster at all.
"""

import re
from pathlib import Path

import pytest
import yaml

K8S_DIR = Path("k8s")
NAMESPACE = "telco-churn"
IMAGE_NAME = "telco-churn-mlops"

WORKLOAD_KINDS = {"Deployment", "Job", "CronJob"}


def load_documents():
    if not K8S_DIR.exists():
        pytest.skip("k8s directory not found")
    docs = []
    for path in sorted(K8S_DIR.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc:
                docs.append((path.name, doc))
    return docs


@pytest.fixture(scope="module")
def documents():
    docs = load_documents()
    if not docs:
        pytest.skip("no manifests found")
    return docs


def pod_specs(documents):
    """Every pod template across Deployments, Jobs and CronJobs."""
    for name, doc in documents:
        kind = doc.get("kind")
        if kind == "Deployment" or kind == "Job":
            yield name, doc, doc["spec"]["template"]["spec"]
        elif kind == "CronJob":
            yield name, doc, doc["spec"]["jobTemplate"]["spec"]["template"]["spec"]


class TestDocumentBasics:
    def test_every_document_is_well_formed(self, documents):
        for name, doc in documents:
            assert "apiVersion" in doc, f"{name}: document missing apiVersion"
            assert "kind" in doc, f"{name}: document missing kind"
            assert doc.get("metadata", {}).get("name"), f"{name}: {doc['kind']} has no name"

    def test_namespaced_resources_declare_the_namespace(self, documents):
        for name, doc in documents:
            if doc["kind"] == "Namespace":
                continue
            assert doc["metadata"].get("namespace") == NAMESPACE, (
                f"{name}: {doc['kind']}/{doc['metadata']['name']} is not in {NAMESPACE}"
            )

    def test_namespace_is_defined(self, documents):
        kinds = {doc["kind"] for _, doc in documents}
        assert "Namespace" in kinds

    def test_expected_workloads_exist(self, documents):
        names = {(d["kind"], d["metadata"]["name"]) for _, d in documents}
        assert ("Deployment", "telco-churn-api") in names
        assert ("Service", "telco-churn-api") in names
        assert ("CronJob", "telco-churn-scoring") in names
        assert ("Job", "telco-churn-train") in names


class TestVolumeWiring:
    """A volume referencing a claim that does not exist fails only at runtime."""

    def test_every_claim_reference_resolves(self, documents):
        declared = {
            doc["metadata"]["name"] for _, doc in documents
            if doc["kind"] == "PersistentVolumeClaim"
        }
        for name, doc, spec in pod_specs(documents):
            for volume in spec.get("volumes", []):
                claim = volume.get("persistentVolumeClaim", {}).get("claimName")
                if claim:
                    assert claim in declared, (
                        f"{name}: volume '{volume['name']}' references unknown claim '{claim}'"
                    )

    def test_every_mount_resolves_to_a_declared_volume(self, documents):
        for name, doc, spec in pod_specs(documents):
            volume_names = {v["name"] for v in spec.get("volumes", [])}
            for container in spec["containers"]:
                for mount in container.get("volumeMounts", []):
                    assert mount["name"] in volume_names, (
                        f"{name}: container '{container['name']}' mounts undeclared "
                        f"volume '{mount['name']}'"
                    )

    def test_claims_request_storage(self, documents):
        for name, doc in documents:
            if doc["kind"] == "PersistentVolumeClaim":
                assert doc["spec"]["resources"]["requests"]["storage"]
                assert doc["spec"]["accessModes"]


class TestConfigMap:
    @pytest.fixture(scope="class")
    def config(self, documents):
        return next(d for _, d in documents if d["kind"] == "ConfigMap")["data"]

    @pytest.mark.parametrize("key", ["DB_URL", "MLFLOW_TRACKING_URI"])
    def test_sqlite_uris_are_relative(self, config, key):
        """sqlite://// is an absolute path the non-root container cannot create.

        Regression test: `sqlite:////data/telco.db` made the training Job die
        with PermissionError trying to mkdir /data at the filesystem root.
        """
        value = config[key]
        if value.startswith("sqlite:"):
            assert not value.startswith("sqlite:////"), (
                f"{key}={value} points at an absolute path; use three slashes"
            )


class TestVolumeShadowing:
    """A volume mounted over a directory hides whatever the image shipped there."""

    def test_no_mount_shadows_the_bundled_raw_data(self, documents):
        """data/raw/Telco-Customer-Churn.csv ships in the image.

        Regression test: mounting an empty PVC at /home/mluser/app/data hid it,
        so the SQL ingest could not find the raw CSV.
        """
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                for mount in container.get("volumeMounts", []):
                    path = mount["mountPath"].rstrip("/")
                    assert path != "/home/mluser/app/data", (
                        f"{name}: mount at {path} shadows the bundled data/raw"
                    )
                    assert path != "/home/mluser/app", (
                        f"{name}: mount at {path} shadows the whole application"
                    )


class TestImageTags:
    def test_no_workload_uses_the_latest_tag(self, documents):
        """`:latest` plus IfNotPresent silently pins pods to a stale image.

        Regression test: the cluster's containerd keeps its own image store, so
        rebuilding `telco-churn-mlops:latest` did not replace what the node had
        cached and pods kept serving the previous build.
        """
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                assert not container["image"].endswith(":latest"), (
                    f"{name}: {container['image']} — use an immutable version tag"
                )

    def test_every_workload_uses_the_same_tag(self, documents):
        images = {
            container["image"]
            for _, _, spec in pod_specs(documents)
            for container in spec["containers"]
        }
        assert len(images) == 1, f"workloads disagree on the image: {sorted(images)}"

    def test_manifest_tag_matches_the_makefile_version(self, documents):
        """A Makefile VERSION bump that misses the manifests deploys nothing new."""
        makefile = Path("Makefile")
        if not makefile.exists():
            pytest.skip("Makefile not found")
        match = re.search(r"^VERSION \?= (.+)$", makefile.read_text(), re.MULTILINE)
        assert match, "Makefile does not define VERSION"
        version = match.group(1).strip()

        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                assert container["image"].endswith(f":{version}"), (
                    f"{name}: {container['image']} does not match "
                    f"Makefile VERSION={version}"
                )


class TestContainers:
    def test_all_workloads_use_the_project_image(self, documents):
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                assert container["image"].startswith(f"{IMAGE_NAME}:"), (
                    f"{name}: unexpected image {container['image']}"
                )

    def test_image_pull_policy_suits_a_local_image(self, documents):
        """`Always` would fail on minikube, where the image is side-loaded."""
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                assert container.get("imagePullPolicy") != "Always", (
                    f"{name}: imagePullPolicy Always cannot work without a registry"
                )

    def test_resources_are_bounded(self, documents):
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                resources = container.get("resources", {})
                assert resources.get("requests"), f"{name}/{container['name']}: no requests"
                assert resources.get("limits"), f"{name}/{container['name']}: no limits"

    def test_containers_run_as_non_root(self, documents):
        for name, doc, spec in pod_specs(documents):
            assert spec.get("securityContext", {}).get("runAsNonRoot") is True, (
                f"{name}: pod does not enforce runAsNonRoot"
            )

    def test_config_is_injected_not_hardcoded(self, documents):
        for name, doc, spec in pod_specs(documents):
            for container in spec["containers"]:
                sources = container.get("envFrom", [])
                assert any(s.get("configMapRef") for s in sources), (
                    f"{name}/{container['name']}: no ConfigMap wired in"
                )


class TestApiDeployment:
    @pytest.fixture(scope="class")
    def deployment(self, documents):
        return next(d for _, d in documents if d["kind"] == "Deployment")

    def test_has_all_three_probes(self, deployment):
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            assert probe in container, f"API container missing {probe}"

    def test_probes_target_the_named_http_port(self, deployment):
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        port_names = {p["name"] for p in container["ports"]}
        for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
            target = container[probe]["httpGet"]["port"]
            assert target in port_names, f"{probe} targets undeclared port {target}"

    def test_readiness_and_liveness_use_different_endpoints(self, deployment):
        """Readiness must gate on the model; liveness must not.

        Regression test: both probes pointed at /health, which returns 200 even
        with no model. Pods were admitted to the Service while unable to serve.
        """
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"
        # Liveness on /ready would crash-loop a pod that is merely awaiting
        # its first trained model.
        assert container["startupProbe"]["httpGet"]["path"] == "/health"

    def test_rollout_preserves_capacity(self, deployment):
        rolling = deployment["spec"]["strategy"]["rollingUpdate"]
        assert rolling["maxUnavailable"] == 0

    def test_artifacts_mounted_read_only(self, deployment):
        """Serving pods must not be able to corrupt the shared artifact volume."""
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        artifacts = next(
            m for m in container["volumeMounts"] if m["name"] == "artifacts"
        )
        assert artifacts.get("readOnly") is True

    def test_replicated(self, deployment):
        assert deployment["spec"]["replicas"] >= 2


class TestService:
    def test_selector_matches_deployment_labels(self, documents):
        """A selector mismatch yields a Service with no endpoints."""
        service = next(d for _, d in documents if d["kind"] == "Service")
        deployment = next(d for _, d in documents if d["kind"] == "Deployment")
        pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
        for key, value in service["spec"]["selector"].items():
            assert pod_labels.get(key) == value, (
                f"Service selector {key}={value} matches no pod label"
            )

    def test_target_port_is_declared_by_the_container(self, documents):
        service = next(d for _, d in documents if d["kind"] == "Service")
        deployment = next(d for _, d in documents if d["kind"] == "Deployment")
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        port_names = {p["name"] for p in container["ports"]}
        for port in service["spec"]["ports"]:
            assert port["targetPort"] in port_names


class TestCronJob:
    @pytest.fixture(scope="class")
    def cronjob(self, documents):
        return next(d for _, d in documents if d["kind"] == "CronJob")

    def test_schedule_has_five_fields(self, cronjob):
        assert len(cronjob["spec"]["schedule"].split()) == 5

    def test_overlapping_runs_are_forbidden(self, cronjob):
        """Two concurrent scoring runs would write two prediction sets per day."""
        assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"

    def test_history_is_bounded(self, cronjob):
        assert cronjob["spec"]["successfulJobsHistoryLimit"] > 0
        assert cronjob["spec"]["failedJobsHistoryLimit"] > 0

    def test_pods_do_not_restart_in_place(self, cronjob):
        spec = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
        assert spec["restartPolicy"] == "Never"
