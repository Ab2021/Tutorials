# Day 66: Model Registry & Artifact Management
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why should you use Safetensors instead of Pickle (PyTorch default)?

**Answer:**
- **Security:** Pickle is insecure by design. Loading a pickle file can execute arbitrary code (e.g., `rm -rf /`). Safetensors is purely data, no code execution.
- **Performance:** Safetensors uses Zero-Copy memory mapping (mmap). It loads much faster, especially for large LLMs, as it doesn't require CPU processing to deserialize objects.
- **Interoperability:** It's language-agnostic (Rust, Python, JS).

#### Q2: What is the role of a Model Registry in MLOps?

**Answer:**
- It serves as the **Single Source of Truth** for model assets.
- It decouples **Training** (Producers) from **Serving** (Consumers).
- It manages **Lifecycle** (Versioning, Stages, Approval).
- It ensures **Governance** (Audit logs of who deployed what).

#### Q3: How do you handle large model artifacts (100GB+)?

**Answer:**
- **Storage:** Use Object Storage (S3) with multipart upload.
- **Transfer:** Use high-bandwidth networks (VPC Endpoints) to avoid public internet costs/slowness.
- **Caching:** Cache weights on the inference node (local disk) to avoid re-downloading on restart.
- **Format:** Use sharded checkpoints (e.g., `model-00001-of-00005.bin`) to parallelize download.

#### Q4: Explain "Blue/Green Deployment" for Models.

**Answer:**
- **Blue:** Current Production version.
- **Green:** New Candidate version.
- **Process:** Deploy Green alongside Blue. Route 0% traffic to Green. Run smoke tests. Gradually shift traffic (1% -> 10% -> 100%). If errors spike, rollback instantly.
- **Registry Role:** Registry tracks which version is Blue and which is Green.

#### Q5: What is Model Lineage?

**Answer:**
- The complete history of how a model was created.
- **Chain:** `Raw Data -> Clean Data -> Training Job (Code + Config) -> Model Artifact -> Deployment`.
- **Importance:** Debugging (Why is it failing?), Compliance (GDPR - what data was used?), and Reproducibility.

---

### Production Challenges

#### Challenge 1: Deployment of Wrong Model

**Scenario:** Intern accidentally promotes a "Test" model to "Production".
**Root Cause:** Lack of permission controls (RBAC) on the Registry.
**Solution:**
- **RBAC:** Only "Lead Engineers" or "CI/CD Bot" can transition to Production.
- **Webhooks:** Alert Slack channel on promotion events.

#### Challenge 2: Slow Startup due to Download

**Scenario:** Autoscaling adds a new node. It takes 10 minutes to download the 70B model from S3.
**Root Cause:** Network bandwidth bottleneck.
**Solution:**
- **Shared Volume:** Mount EFS/NFS (faster than S3 download if provisioned correctly).
- **Image Baking:** Bake model into Docker image (fastest startup, slow build).
- **P2P:** Use peer-to-peer distribution (Dragonfly) for large clusters.

#### Challenge 3: Corrupted Artifacts

**Scenario:** Model loads but outputs garbage.
**Root Cause:** Upload was interrupted or file corruption.
**Solution:**
- **Checksums:** Verify MD5/SHA256 hash after download.
- **Atomic Uploads:** Upload to a temporary path, then rename.

#### Challenge 4: Registry Sync Issues

**Scenario:** Registry says "v5 is Prod", but API is serving "v4".
**Root Cause:** Serving infrastructure didn't pick up the change.
**Solution:**
- **Push-based:** CI/CD pipeline forces a redeploy of the serving service.
- **Health Check:** API endpoint `/health/version` should return the currently loaded version. Monitor for drift.

#### Challenge 5: Dependency Hell

**Scenario:** Model v1 used `transformers==4.30`. Model v2 uses `transformers==4.35`. Serving container has `4.30`. v2 fails.
**Root Cause:** Model artifacts are coupled with code/environment.
**Solution:**
- **Container Registry:** Version the *Container Image*, not just the *Model Weights*.
- **Metadata:** Store `requirements.txt` in the Model Registry alongside weights.

### System Design Scenario: Secure Model Supply Chain

**Requirement:** Bank needs to ensure no malicious models are deployed.
**Design:**
1.  **Training:** In secure VPC.
2.  **Signing:** CI/CD pipeline signs the model artifact with a Hardware Security Module (HSM) key.
3.  **Registry:** Stores signed model + signature.
4.  **Scanning:** Artifacts are scanned for malware/pickles.
5.  **Deployment:** Inference server has the Public Key. Verifies signature before `load_state_dict`. Refuses to load if verification fails.

### Summary Checklist for Production
- [ ] **Format:** Use **Safetensors**.
- [ ] **Storage:** Use **S3** with versioning enabled.
- [ ] **Security:** Implement **RBAC** for promotion.
- [ ] **Validation:** Run **Automated Tests** before promotion.
- [ ] **Cleanup:** Automate **Old Artifact Deletion**.
