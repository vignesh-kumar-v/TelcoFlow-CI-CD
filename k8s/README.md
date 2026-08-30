# Kubernetes deployment

Runs the churn API as a replicated Deployment behind a Service, with batch
training and nightly scoring as Jobs sharing a persistent artifacts volume.

```
┌──────────────────┐        ┌─────────────────────┐
│ Job: train       │ writes │ PVC: telco-artifacts│
│ (sql + train)    ├───────>│  model.joblib       │
└──────────────────┘        │  train_info.json    │
                            └──────────┬──────────┘
┌──────────────────┐  reads            │
│ CronJob: scoring │<──────────────────┤
│ (nightly 02:00)  │                   │
└──────────────────┘                   │ reads (ro)
                            ┌──────────▼──────────┐
                            │ Deployment: api x2  │<── Service :80
                            └─────────────────────┘
```

## Local run

There is no registry, so the cluster has to see a locally built image
(`imagePullPolicy: IfNotPresent`). How you achieve that is the only difference
between the two options.

### Option A — Docker Desktop Kubernetes (simplest)

Enable it under **Settings → Kubernetes → Enable Kubernetes**. The cluster
shares Docker Desktop's image store, so a normal build is immediately visible
to it — no loading step:

```bash
make docker-build     # tags telco-churn-mlops:$(VERSION) and :latest
kubectl config use-context docker-desktop
```

### Option B — minikube

minikube runs its own Docker daemon, so the image must be built inside it (or
side-loaded with `minikube image load telco-churn-mlops:0.2.0`):

```bash
minikube start --memory=4096
eval $(minikube docker-env)
make docker-build
```

The tag matters: the manifests pin `telco-churn-mlops:0.2.0`, so an image
tagged only `:latest` leaves the pods in `ImagePullBackOff`. `make docker-build`
applies both tags from the Makefile's `VERSION`.

### Then, for either option

```bash
kubectl apply -f k8s/base.yaml
kubectl apply -f k8s/jobs.yaml
kubectl apply -f k8s/api-deployment.yaml

# Populate the artifacts volume before the API can serve.
kubectl wait --for=condition=complete job/telco-churn-train -n telco-churn --timeout=15m

kubectl get pods -n telco-churn
kubectl port-forward -n telco-churn svc/telco-churn-api 8000:80
curl localhost:8000/health
```

Or use the Make targets: `make k8s-deploy`, `make k8s-status`, `make k8s-delete`.

Give the cluster at least 4GB: the training Job requests 1Gi and may burst to
4Gi.

## Design notes

**Liveness and readiness are different endpoints, deliberately.**

| Probe | Endpoint | Behaviour |
|---|---|---|
| startup, liveness | `/health` | 200 whenever the process is up, model or not |
| readiness | `/ready` | 503 until a model is loaded |

Splitting them matters because probes inspect the *status code*, not the body.
An earlier version pointed readiness at `/health`, which returned 200 with
`{"status": "no_model"}` — Kubernetes marked those pods `1/1 READY` and added
them to the Service, so they received traffic they could only fail. Verified
against a live cluster: pods now sit at `0/1` with no Service endpoints until a
model exists.

Liveness must *not* gate on the model. Restarting a container cannot produce a
model that was never trained, so a model-aware liveness probe would crash-loop
a pod that is simply waiting for its first training run.

`/ready` also retries the load, so API pods that start in parallel with the
training Job become ready on their own — observed going `0/1 → 1/1` with **zero
restarts** once training finished.

**`maxUnavailable: 0`** keeps full capacity during rollouts, since the artifact
volume is ReadWriteOnce and a new pod may briefly contend with an old one.

**Storage is the real constraint.** ReadWriteOnce works on single-node minikube
because every pod lands on the same node. On a multi-node cluster the API
replicas and the Jobs can be scheduled apart, and RWO cannot be mounted from
two nodes at once — switch to ReadWriteMany (NFS/EFS/Filestore) or, better,
have training publish to the MLflow registry or object storage and let the API
pull from there instead of sharing a filesystem.

**`concurrencyPolicy: Forbid`** stops a slow scoring run from overlapping the
next night's, which would otherwise write two prediction sets for one day.

**Never `:latest`.** The cluster's containerd keeps its own image store,
separate from Docker's. With `:latest` and `imagePullPolicy: IfNotPresent`, a
rebuilt image is silently ignored and pods keep running the cached one — which
cost real debugging time here. Manifests pin an immutable tag; bump `VERSION`
in the Makefile and the tag in `k8s/*.yaml` together (a test enforces they
match).

**Volumes are mounted at `data/processed`, not `data/`.** The raw CSV ships
inside the image at `data/raw/`, and a volume mounted over `data/` hides it,
breaking the SQL ingest.

Containers run as non-root with all capabilities dropped. `readOnlyRootFilesystem`
is left off because matplotlib and joblib write to temp paths during SHAP
plotting and model loading.
