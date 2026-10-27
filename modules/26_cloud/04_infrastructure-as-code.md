# Infrastructure as Code

> Module 26 — Cloud, Kubernetes, and Streaming Production Labs
> Topic 04 of 09
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

Cloud infrastructure created by hand is a set of undocumented decisions. A subnet was widened
during an incident, a security group rule was added for a demo and never removed, a bucket was
created in the wrong account, and nobody can say which of the three environments matches the
architecture diagram. Infrastructure as code (IaC) exists to make those decisions reviewable,
repeatable, and reversible.

The reference stack in `labs/cloud/terraform/` is small enough to read in an hour. It composes three
modules: `modules/network` for the virtual private cloud (VPC), subnets, and network address
translation (NAT); `modules/storage` for the key management service (KMS) key, the S3 bucket, and
the Elastic Container Registry (ECR) repository; and `modules/eks` for the Elastic Kubernetes
Service cluster and managed node group. The root `main.tf` wires them together, and
`variables.tf` exposes the knobs. Reading it end to end is the fastest way to see how a cloud
environment is actually assembled.

The topic is not about Terraform syntax. It is about four operational properties that IaC either
gives you or does not:

| Property | What it means | What breaks it |
|---|---|---|
| Reproducibility | The same code produces the same environment | Unpinned providers, manual console changes |
| Reviewability | A change is visible before it is applied | Applying from a laptop without a plan review |
| Recoverability | A lost environment can be rebuilt | State stored only on one machine |
| Attribution | Every resource has an owner and a cost centre | Missing tags, shared accounts |

### 1.1 The failures that motivate the discipline

A team creates a development environment by hand. Six months later, a security review asks whether
the S3 bucket blocks public access. Nobody knows, because the bucket was created in the console and
the setting was changed twice. The team enables the block, which breaks a partner integration that
depended on a public prefix. The incident is not caused by the setting; it is caused by the absence
of a record of why the setting was what it was.

A second team stores Terraform state on a engineer's laptop. The engineer leaves. The state is
lost. The infrastructure still runs, but no plan can be trusted, because Terraform believes the
resources do not exist and proposes to create duplicates. Recovery requires importing every
resource by hand, which is possible and slow and error-prone.

A third team applies changes from three laptops against the same state file. Two applies overlap.
The state is corrupted, and the resources are in a state that neither apply intended.

All three are state problems, not syntax problems. That is why Section 5 starts with the backend.

### 1.2 Why this matters in a regulated setting

Configuration management is a required process in medical device software. IEC 62304 clause 8
requires identification, control, and status accounting of configuration items, and clause 8.2
requires change control. An IaC repository with reviewed pull requests, tagged releases, and a
recorded plan is a natural configuration management system. A console-driven environment is not.

The Health Insurance Portability and Accountability Act (HIPAA) Security Rule does not mention IaC. It does require audit controls at 45 CFR 164.312(b)
and integrity protection at 164.312(c)(1). CloudTrail records API calls, including those made by
Terraform, so the combination of IaC and CloudTrail gives a change history that a console-only
workflow also has, but without the intended-state record that explains why the change was made.

> **Note:** The lab stack is a teaching scaffold. Its S3 backend is an example file, its secrets
> are placeholders, and it has not been applied in the authoring environment. It is not a compliant
> deployment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Configure** a remote state backend with encryption, versioning, and locking, and explain what
   each provides.
2. **Pin** provider and module versions and state the consequence of a floating constraint.
3. **Compute** a subnet plan from a virtual private cloud (VPC) CIDR using `cidrsubnet`, and detect overlap between
   environments.
4. **Read** a `terraform plan` and identify which changes force replacement and why.
5. **Estimate** the monthly cost of an environment from its instance types, NAT gateway count, and
   storage, at a stated region and month.
6. **Diagnose** a state lock, a drift report, and an unintended replacement.
7. **Design** an environment promotion path from development to production without copying state.
8. **Name** what Terraform state contains that must not be committed, and how to protect it.

Outcomes 1 and 2 are assessed by Exercises 12.1 and 12.2. Outcomes 3 and 5 by 12.3 and 12.4.
Outcomes 4 and 6 by 12.5 and 12.6. Outcomes 7 and 8 by 12.7 and 12.8.

## 3. The core mental model

Terraform has three artefacts: the configuration, the state, and the provider's remote objects.
The configuration is what you want. The state is what Terraform believes exists. The remote objects
are what actually exists. Every Terraform operation is a reconciliation between the three.

```text
   configuration (git)            state (remote backend)         remote objects (AWS)
   ┌──────────────────┐          ┌────────────────────┐         ┌──────────────────┐
   │ resource blocks  │          │ resource addresses │         │ VPC, subnets     │
   │ module calls     │          │ attribute values   │         │ S3 bucket + KMS  │
   │ variables        │          │ dependency graph   │         │ EKS cluster      │
   └────────┬─────────┘          └─────────┬──────────┘         └────────┬─────────┘
            │                              │                             │
            └──────────────┬───────────────┴──────────────┬──────────────┘
                           ▼                              ▼
                    terraform plan                 terraform apply
                    (diff config vs state,         (execute the plan,
                     then state vs remote)         write new state)
```

The plan is computed in two steps that are easy to conflate. First Terraform refreshes the state
against the remote objects, discovering drift. Then it compares the refreshed state with the
configuration and proposes changes. A plan that says "no changes" means the remote objects match
the configuration, not that nobody changed anything outside Terraform; a refresh would have shown
that.

### 3.1 State is the source of truth, and it is sensitive

State maps resource addresses to remote identifiers and caches attribute values. Some of those
values are secrets: database passwords, generated keys, transport layer security (TLS) private keys. State is therefore a
secret store by accident, and must be protected as one.

| State property | Why it matters | Control |
|---|---|---|
| Contains attribute values | May include passwords and keys | Encrypt at rest, restrict read access |
| Single writer | Concurrent applies corrupt it | Locking |
| Historical | Needed for rollback and audit | Versioning |
| Large | Grows with resource count | Split by environment and component |
| Addressable | Moving a resource changes its address | `moved` blocks |

### 3.2 Modules are interfaces

A module is a boundary with inputs and outputs. `labs/cloud/terraform/modules/network` takes
`project`, `env`, and `vpc_cidr`, and returns `vpc_id` and `private_subnet_ids`. The root module does
not know how many subnets exist or which availability zones were selected. That is the point: the
network module can change internally without changing its callers.

A module that reaches into another module's resources is not a module; it is a folder. Terraform
enforces this through explicit outputs, and reviewers should enforce it socially.

### 3.3 Environments are separate states, not workspaces with identical code

There are two common patterns. Workspaces use one configuration and multiple state files, with
variables differing per workspace. Separate directories use one configuration per environment,
often calling the same modules. The lab uses a single root with an `env` variable, which is closer
to the workspace pattern.

| Pattern | Advantage | Risk |
|---|---|---|
| Workspaces | One code path, less duplication | A variable default can silently apply to production |
| Separate directories | Clear blast radius, independent version pinning | Duplication and drift between directories |
| Separate repositories | Strongest isolation | Heavy promotion overhead |

For a small team, separate directories with a shared module source is the usual compromise. The
critical property is that production state is never in the same backend key as development state.

### 3.4 The plan is the review artifact

A pull request shows code. A plan shows consequences. The review should require both, because a
three-line change can destroy a database. Terraform marks replacement with `-/+` or
`# forces replacement`, and that marker is the single most important thing to read.

```text
  # aws_vpc.this must be replaced
-/+ resource "aws_vpc" "this" {
      ~ cidr_block = "10.42.0.0/16" -> "10.43.0.0/16" # forces replacement
    }
```

A CIDR change on an existing VPC replaces the VPC, which replaces subnets, which replaces the
cluster. The plan will show the whole chain if the dependencies are expressed. That is why the plan
must be reviewed in full, not skimmed for the resource that was edited.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| IaC | Infrastructure as code | A script that calls the cloud API | Practice |
| Provider | Plugin that maps Terraform resources to an API | A module | Plugin version |
| Resource | One managed object | A module | Address |
| Data source | Read-only lookup performed during plan | A resource | Address |
| Module | Reusable group of resources with inputs and outputs | A directory | Source and version |
| Root module | The configuration you run Terraform in | A child module | Directory |
| State | Terraform's record of managed objects | The configuration | JSON, encrypted |
| Backend | Where state is stored and locked | A provider | Configuration block |
| Lock | Mechanism preventing concurrent writes | A mutex in the process | Lock file or table |
| Plan | Proposed changes | The apply | Human-readable diff |
| Apply | Execution of a plan | A plan | Operation |
| Refresh | Reconciliation of state with remote objects | A plan | Operation |
| Drift | Difference between state and remote objects | A configuration change | Attribute diff |
| Import | Adopting an existing object into state | Creating a resource | Operation |
| `moved` block | Instruction to change a resource address without recreation | A rename in code | Block |
| Taint | Marking a resource for replacement | A failed apply | State flag |
| Workspace | Named state within one configuration | An environment | Name |
| Variable | Input to a module | A constant | Type and default |
| Output | Value exported by a module | A log line | Type |
| Locals | Computed values within a module | Variables | Expression |
| `for_each` | Creating one instance per collection member | A loop | Expression |
| `count` | Creating N identical instances | A loop with meaning | Expression |
| Provisioner | Imperative action during apply | A resource | Discouraged |
| Destroy | Removal of managed objects | Deletion of code | Operation |
| Blast radius | Scope affected by one apply | Any change | Description |
| Tag | Key-value metadata for cost and policy | Decoration | Key-value |
| State lock file | S3 object used for native locking | A DynamoDB table | `.tflock` object |
| Remote backend | Backend not on the local filesystem | Any backend | Configuration |
| Policy as code | Automated checks on plans | A wiki page | Tool-specific |

Three distinctions carry most of the operational weight.

**State is not configuration.** Deleting a resource block proposes destruction of the remote
object. Deleting the state file proposes creation of a duplicate. Both are dangerous in different
ways.

**Plan is not apply.** A plan can be stale by the time it is applied. The longer the gap, the more
likely the assumptions changed. Apply the plan that was reviewed, promptly.

**Import is not adoption.** `terraform import` records an object's identifier but does not generate
configuration. The configuration must be written to match, or the next plan will propose changes.

## 5. The mechanics, step by step

The procedure takes an empty cloud account and produces three environments that can be reviewed,
reproduced, and destroyed. It follows the file layout in `labs/cloud/terraform/`.

### Step 1 — Define environments and state boundaries

**Input.** The environments required and their isolation requirements.

**Output.** One state key per environment, and a rule for what shares a state file.

**Rationale.** The state file is the unit of blast radius. A single state for all environments means
a mistyped apply can affect production. Separate keys mean a development mistake stays in
development.

**Branch point.** If environments share a network (a hub-and-spoke topology), the shared network has
its own state, and the environment states read its outputs through remote state data sources. Do
not put shared and environment-specific resources in one state.

### Step 2 — Configure the remote backend with locking

**Input.** An S3 bucket for state, a key prefix, and a region.

**Output.** A `backend "s3"` block with encryption, versioning, and locking enabled.

**Rationale.** Remote state gives durability, access control, and a single writer. Locking prevents
concurrent applies. Encryption protects the secrets that state accumulates.

```hcl
terraform {
  backend "s3" {
    bucket       = "biomed-tfstate-123456789012"
    key          = "prod/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}
```

`use_lockfile` enables S3-native locking through a `.tflock` object next to the state. It became
generally available in Terraform 1.11.0 in February 2025. The older `dynamodb_table` argument is
deprecated and will be removed in a future minor version, though it still works and can be
configured alongside `use_lockfile` during a migration.

The state bucket itself must be created outside this configuration, or the configuration would
depend on state that does not exist yet. That bootstrap step is a manual or separately applied
root, and it must have versioning and public access blocked.

**Edge case.** If the bucket is in another account, the Terraform execution role needs
`s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject` on the state key and the lock object. Missing
delete permission produces a lock that is never released.

### Step 3 — Pin provider and module versions

**Input.** The cloud provider and the modules in use.

**Output.** A `versions.tf` with explicit constraints for Terraform, providers, and modules.

**Rationale.** An unpinned provider can change resource behaviour between runs. An unpinned module
can change its inputs or defaults. `labs/cloud/terraform/versions.tf` and the module blocks pin
major versions; the EKS module uses `~> 20.0` and the VPC module uses `~> 5.0`.

**Branch point.** A pessimistic constraint such as `~> 20.0` allows minor upgrades within major
version 20 and blocks 21. That is a reasonable default. For a regulated environment, pin exactly
and upgrade deliberately.

**Edge case.** The Terraform Registry's current EKS module release line is 21.x, with 21.29.0
published on 7 October 2026 and requiring Terraform 1.5.7 or later and AWS provider 6.59 or later.
The lab's `~> 20.0` is a major version behind. Upgrading is not a version bump; the v21 line changed
authentication defaults and added features such as access entries and Auto Mode. Read the migration
guide.

### Step 4 — Compose the root module

**Input.** The module sources and the variables.

**Output.** A root `main.tf` that passes outputs of one module into inputs of another.

**Rationale.** `labs/cloud/terraform/main.tf` passes `module.network.vpc_id` and
`module.network.private_subnet_ids` into the EKS module. That edge is what makes Terraform create
the network before the cluster and destroy the cluster before the network. The dependency graph is
derived from references, not from file order.

```hcl
module "network" {
  source   = "./modules/network"
  project  = var.project
  env      = var.env
  vpc_cidr = var.vpc_cidr
}

module "eks" {
  source              = "./modules/eks"
  project             = var.project
  env                 = var.env
  cluster_version     = var.cluster_version
  vpc_id              = module.network.vpc_id
  private_subnet_ids  = module.network.private_subnet_ids
  node_instance_types = var.node_instance_types
}
```

**Edge case.** Implicit dependencies are better than `depends_on`. A reference creates a precise
edge; `depends_on` creates a coarse one that can serialise unrelated work and slow the apply.

### Step 5 — Build the network module

**Input.** A VPC CIDR and an environment name.

**Output.** A VPC, three private subnets, three public subnets, a NAT gateway, and route tables.

**Rationale.** `labs/cloud/terraform/modules/network/main.tf` selects the first three available
availability zones and calls `cidrsubnet` with a four-bit extension, producing sixteen /20 networks
from a /16. It uses indices 0, 1, 2 for private subnets and 8, 9, 10 for public, leaving a gap for
future use.

```text
10.42.0.0/16
├── 10.42.0.0/20     private, zone a
├── 10.42.16.0/20    private, zone b
├── 10.42.32.0/20    private, zone c
├── 10.42.48.0/20    reserved
├── ...
├── 10.42.128.0/20   public, zone a
├── 10.42.144.0/20   public, zone b
└── 10.42.160.0/20   public, zone c
```

Each /20 provides 4,091 usable addresses, which is ample for a lab and modest for a production
cluster running the VPC CNI, where each pod consumes an address.

**Branch point.** `single_nat_gateway = true` creates one NAT gateway for all private subnets. It
saves money and creates a single point of failure for outbound traffic. Set it to false in
production and accept three NAT gateways.

**Edge case.** Overlapping CIDRs across environments prevent peering and make hybrid connectivity
impossible. Allocate from a plan, not ad hoc. Section 7 includes a check.

### Step 6 — Build the storage module

**Input.** A project name and an environment.

**Output.** A KMS key and alias, a versioned and encrypted S3 bucket with public access blocked, a
lifecycle rule, and an immutable Elastic Container Registry (ECR) repository with scan on push.

**Rationale.** Each resource in `labs/cloud/terraform/modules/storage/main.tf` closes a specific
failure. Versioning protects against overwrite and deletion. Server-side encryption with a customer
managed key gives key-level access control and an audit trail in CloudTrail. The public access
block prevents the most common data exposure. The lifecycle rule expires noncurrent versions after
30 days, which bounds storage cost.

```hcl
resource "aws_s3_bucket_public_access_block" "raw" {
  bucket                  = aws_s3_bucket.raw.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

**Branch point.** A customer managed key adds a monthly key charge and per-request charges, and
adds an operational dependency: if the key is disabled, the data is unreadable. An AWS managed key
is cheaper and less controllable. For protected health information (PHI), the customer managed key is usually the right choice
because it supports key policies and revocation.

**Edge case.** `deletion_window_in_days = 7` is the minimum. A deleted key cannot be recovered after
the window. Accidental key deletion makes every object encrypted with it permanently unreadable,
even if the bucket is intact.

### Step 7 — Build the EKS module

**Input.** The VPC, private subnet identifiers, a cluster version, and node group sizing.

**Output.** An EKS cluster, a managed node group, and outputs for the cluster endpoint and name.

**Rationale.** `labs/cloud/terraform/modules/eks/main.tf` pins the community module at `~> 20.0`,
enables IAM roles for service accounts (IRSA), and creates one managed node group of `t3.medium`
instances with desired size 2, minimum 1, and maximum 4.

**Branch point.** `cluster_endpoint_public_access = true` makes the API server reachable from the
internet. It is convenient for a lab and is usually unacceptable for production. The production
setting is private access with a bastion or a VPN, or public access restricted to known CIDRs.

**Edge case.** A managed node group does not upgrade when the control plane is upgraded. The
Terraform change that bumps `cluster_version` updates the control plane and leaves the nodes on the
old version. A separate node group update is required, and the plan should show it.

### Step 8 — Validate variables and defaults

**Input.** The variables exposed by the root module.

**Output.** Types, defaults, descriptions, and validation rules.

**Rationale.** `labs/cloud/terraform/variables.tf` validates that `env` is one of `dev`, `staging`,
or `prod`. Validation prevents a typo from creating a fourth environment that nobody monitors.

**Branch point.** If a variable has a default that is safe only for development, do not give it a
default. Make the caller choose. Defaults propagate silently into production.

### Step 9 — Review the plan

**Input.** A branch with the proposed change and credentials for the target account.

**Output.** A plan file, a human review, and a recorded approval.

**Rationale.** The plan is the only artefact that shows consequences. Save it with
`terraform plan -out=tfplan` and apply that exact file, so what was reviewed is what runs.

**Branch point.** If the plan shows a replacement of a stateful resource, stop. Replacements of
databases, buckets, and KMS keys are data-loss events unless a migration is planned. Mark the
resource with `lifecycle { prevent_destroy = true }` where appropriate.

**Edge case.** A plan can be huge. Review the summary counts first, then the resources with
`forces replacement`, then everything else. A plan with no replacements and only in-place updates
is usually safe to read quickly.

### Step 10 — Apply with an approval gate

**Input.** The saved plan and an execution identity.

**Output.** Applied infrastructure and an updated state.

**Rationale.** The apply identity should be a role assumed by a continuous integration system, not
a human's administrator credentials. The role's permissions should be the minimum needed for the
resources in the configuration.

**Branch point.** If the apply fails partway, the state contains what succeeded. Do not re-run
blindly; read the error, fix the cause, and re-plan. Terraform is idempotent for what it created,
but a partial apply can leave dependent resources in an intermediate state.

### Step 11 — Detect and reconcile drift

**Input.** A scheduled plan against each environment.

**Output.** A drift report and a decision for each difference.

**Rationale.** Console changes, auto-scaling, and provider-side updates all create drift. A daily
plan that reports no changes is evidence that the environment matches the code. A plan that reports
changes is either a pending code change or an unauthorised manual change.

**Branch point.** Some drift is expected: an auto-scaling group's desired count, a managed node
group's current size. Exclude those attributes from the comparison, or accept them in the review.
Do not disable drift detection entirely because one attribute is noisy.

### Step 12 — Destroy safely

**Input.** An environment scheduled for removal.

**Output.** Removed resources, retained data where required, and an empty state.

**Rationale.** `terraform destroy` removes what the state knows about. It does not remove resources
created outside Terraform, and it does remove data unless deletion protection is set.

**Branch point.** For a production environment, the destroy path should require a separate
approval and should retain the S3 bucket and KMS key. `lifecycle { prevent_destroy = true }` makes
the destroy fail loudly rather than silently deleting data.

**Edge case.** A VPC with a NAT gateway, an elastic network interface, or a load balancer can fail
to delete because the dependency is not modelled. The error names the resource; the fix is to model
it, not to delete it by hand.

### 5.1 Edge cases that change the answer

**State lock contention.** Two applies start at once. The second fails with
`Error acquiring the state lock`. If the first is still running, wait. If it crashed, the lock
record remains and must be released with `terraform force-unlock <lock-id>` after confirming no
apply is active. Force-unlocking a live apply corrupts state.

**Credential expiry.** A long apply that assumes a role with a one-hour session can fail midway when
the credentials expire. The state records the resources created so far. Re-authenticate and re-plan;
do not retry the same process with stale environment variables.

**Quota exhaustion.** A new environment requests more vCPUs than the account quota allows. The
apply fails on the node group or the instances never join. The error may appear as a failed
autoscaling group rather than a quota message. Check the Service Quotas console for the instance
family.

**Provider upgrade.** A provider major version changes resource schemas. The plan after an upgrade
may show replacements that were not intended. Read the provider's upgrade guide before changing the
constraint.

**State loss.** The state bucket is deleted or the key is wrong. Terraform proposes to create
everything. Do not apply. Restore the state from a versioned copy, or import the existing resources.

**Secrets in state.** A database resource writes its generated password into state. Anyone with
read access to the state bucket can read it. Restrict access, enable encryption, and rotate
credentials that were exposed.

**Resource moved without a `moved` block.** Renaming a resource address in code makes Terraform
destroy the old object and create a new one. A `moved` block tells Terraform to update the address
instead. This is the difference between a rename and an outage.

### 5.2 A decision table for environment isolation

| Requirement | Workspaces | Separate directories | Separate repositories |
|---|---|---|---|
| One code path | Yes | Duplicated | Duplicated |
| Independent state | Yes | Yes | Yes |
| Independent version pinning | No | Yes | Yes |
| Clear blast radius | Weak | Strong | Strongest |
| Promotion is a reviewed diff | No | Partial | Partial |
| Suitable for regulated production | Not alone | Yes | Yes |

### 5.3 A module interface checklist

A module is only reusable if its interface is honest. Before publishing one internally, check:

| Check | Why | Failure if skipped |
|---|---|---|
| Every input has a type and description | Callers need to know what to pass | Stringly typed configuration |
| Every output is something a caller needs | Outputs are the contract | Callers reach into internals |
| No provider configuration inside the module | Providers are a root concern | Two providers, unpredictable behaviour |
| No hard-coded region or account | Portability | Cannot deploy to a second region |
| Tags accepted as a map and merged | Cost attribution | Untagged resources |
| Resource names derived from inputs | Predictable naming | Collisions between environments |
| `prevent_destroy` on stateful resources | Accidental deletion | Data loss |
| A README with an example call | Usability | Copy-paste from another module |
| An output for every value a caller needs | Composition | Duplicated resources |

The lab modules follow most of these. `modules/network` and `modules/storage` take `project` and
`env`; `modules/eks` takes the VPC outputs rather than looking them up. None of them configures a
provider, which is why the root module can set the region once.

## 6. Worked example

A team runs three environments from the same module set: development, staging, and production. The
question is what the environments cost, whether the network plan is sound, and what happens when a
CIDR changes.

### 6.1 Inputs

| Parameter | Value | Source |
|---|---|---|
| Region | us-east-2 (Ohio) | `var.aws_region` default is us-east-1; this example uses Ohio for price consistency |
| Currency and month | USD, 730 hours | Convention |
| EKS control plane | USD 0.10 per cluster-hour | AWS Price List, us-east-2, October 2026 |
| NAT gateway | USD 0.045 per gateway-hour | AWS VPC pricing example, us-east-2 |
| `t3.medium` | USD 0.0416 per hour, 2 vCPU, 4 GiB | AWS Price List, us-east-2, October 2026 |
| `m7g.xlarge` | USD 0.1632 per hour, 4 vCPU, 16 GiB | AWS Price List, us-east-2, October 2026 |
| S3 Standard | USD 0.023 per GB-month, first 50 TB | AWS S3 pricing, us-east-2 |
| gp3 | USD 0.08 per GB-month | AWS EBS pricing example rate |
| VPC CIDRs | dev 10.42.0.0/16, staging 10.43.0.0/16, prod 10.44.0.0/16 | Proposed |

### 6.2 Arithmetic: subnet plan

`cidrsubnet("10.42.0.0/16", 4, i)` extends the prefix by four bits, from /16 to /20, producing
sixteen subnets. Private subnets use indices 0 to 2, public subnets use 8 to 10:

```text
private: 10.42.0.0/20, 10.42.16.0/20, 10.42.32.0/20
public:  10.42.128.0/20, 10.42.144.0/20, 10.42.160.0/20
```

Each /20 has 2^(32−20) = 4,096 addresses, of which 4,091 are usable in AWS because AWS reserves
five per subnet.

### 6.3 Arithmetic: development environment

```text
EKS             1 × 0.10   × 730 =  73.00
Nodes           2 × 0.0416 × 730 =  60.74
NAT             1 × 0.045  × 730 =  32.85
S3              100 GB × 0.023   =   2.30
gp3             100 GB × 0.08    =   8.00
                                   ------
total                             176.89 USD/month
```

### 6.4 Arithmetic: production environment

```text
EKS             1 × 0.10   × 730 =  73.00
Nodes           6 × 0.1632 × 730 = 714.82
NAT             3 × 0.045  × 730 =  98.55
S3              500 GB × 0.023   =  11.50
gp3             600 GB × 0.08    =  48.00
                                   ------
total                             945.87 USD/month
```

### 6.5 Arithmetic: the three-environment total

```text
dev      176.89
staging  187.19
prod     945.87
        -------
total   1,309.94 USD/month
```

The production environment is 72 percent of the cost, and node compute is 76 percent of production.
That is the right shape: the control plane is cheap, the network is cheap, and the compute is where
the money is. A cost review should start with node sizing, not with storage.

### 6.6 Arithmetic: the NAT gateway trade-off

Production with one NAT gateway instead of three:

```text
1 × 0.045 × 730 = 32.85
945.87 − 98.55 + 32.85 = 880.17 USD/month
```

The three-zone premium is:

```text
98.55 − 32.85 = 65.70 USD/month
```

That is the price of removing a single point of failure for all outbound traffic. It is 7 percent
of the production bill and is usually worth paying. The lab sets `single_nat_gateway = true` for
the opposite reason.

### 6.7 Two diagnostics that reveal a misconfiguration

**Diagnostic 1: the plan proposes to replace the bucket.**

```bash
terraform plan -out=tfplan
terraform show tfplan | grep -B2 -A2 "forces replacement"
```

If the output shows `aws_s3_bucket.raw must be replaced`, the cause is usually a changed bucket
name. The lab derives the name from `random_id.suffix`, so a lost state produces a new random suffix
and a new bucket. The old bucket still exists with the data, and the new bucket is empty. The check
is `terraform state list` to confirm whether the resource is in state at all. If it is not, restore
the state before applying.

**Diagnostic 2: the apply fails with a state lock.**

```bash
terraform apply
# Error: Error acquiring the state lock
```

The error includes a lock identifier, who created it, and when. Confirm whether that process is
still running. If it is, wait. If it is not, release it:

```bash
terraform force-unlock <lock-id>
```

Force-unlocking while another apply is running causes two writers to update the same state. That is
the failure the lock exists to prevent, and recovering from it means reconstructing state from the
remote objects.

A third diagnostic worth knowing is a plan that proposes to destroy and recreate the EKS cluster
after a version change. The control plane version is an in-place update in most cases, but the node
group's launch template may be replaced, which replaces the nodes. The plan shows this as a
replacement of the node group, not of the cluster. Draining and rolling is the correct procedure;
applying the replacement during clinical hours is not.

A fourth diagnostic is a plan that contains a resource count far larger than the change. A
one-line edit that produces a 40-resource plan usually means a module input changed in a way that
propagates, for example an environment tag applied to every resource. That is sometimes intended.
Confirm by grouping the plan by module and resource type before applying:

```bash
terraform show -json tfplan | jq -r '
  [.resource_changes[] | select(.change.actions != ["no-op"])]
  | group_by(.module_address // "root")
  | map({module: .[0].module_address // "root", count: length})
  | .[] | "\(.count)\t\(.module)"'
```

### 6.8 Vary one assumption

Change production node type from `m7g.xlarge` to `m7i.xlarge`. The node count stays at six and the
hourly price rises from USD 0.1632 to USD 0.2016:

```text
6 × 0.2016 × 730 = 883.01
945.87 − 714.82 + 883.01 = 1,114.06 USD/month
```

The difference is USD 168.19 per month, or about 18 percent of the production bill, for the same
vCPU and memory. The Graviton instance is cheaper. The reason to choose `m7i` is compatibility: an
x86-only binary, a vendor agent, or a native library without an arm64 build. That is a real reason,
and it should be written down, because the default should be the cheaper instance when nothing
requires otherwise.

A second variation is the node count. If the production workload fits on four nodes rather than six,
the production bill falls by:

```text
2 × 0.1632 × 730 = 238.27 USD/month
945.87 − 238.27 = 707.60 USD/month
```

The saving is larger than the instance-type change, which is the general pattern: node count is the
dominant cost lever, and instance family is second. Both are set by the capacity model in
[`03_kubernetes.md`](03_kubernetes.md), not by the Terraform code. That is why the two topics are
adjacent.

## 7. Code, query, or calculation

The following standard-library Python script implements the `cidrsubnet` arithmetic used by the
network module, checks for overlap between environment CIDRs, and computes monthly cost per
environment. It reproduces Section 6.

```python
"""CIDR plan and monthly cost model for the reference Terraform stack. Stdlib only."""
from __future__ import annotations

import ipaddress
import json
from dataclasses import dataclass


@dataclass
class Environment:
    name: str
    vpc_cidr: str
    az_count: int
    nat_gateways: int
    node_type: str
    node_count: int
    node_hourly: float
    s3_gb: int
    gp3_gb: int


HOURS = 730.0
EKS_HOURLY = 0.10          # per cluster, us-east-2, October 2026
NAT_HOURLY = 0.045         # per NAT gateway, us-east-2, October 2026
S3_GB_MONTH = 0.023
GP3_GB_MONTH = 0.08

ENVIRONMENTS = [
    Environment("dev", "10.42.0.0/16", 3, 1, "t3.medium", 2, 0.0416, 100, 100),
    Environment("staging", "10.43.0.0/16", 3, 1, "t3.medium", 2, 0.0416, 200, 200),
    Environment("prod", "10.44.0.0/16", 3, 3, "m7g.xlarge", 6, 0.1632, 500, 600),
]


def subnets(vpc_cidr: str, new_prefix: int) -> list[str]:
    network = ipaddress.ip_network(vpc_cidr)
    return [str(subnet) for subnet in network.subnets(new_prefix=new_prefix)]


def monthly_cost(env: Environment) -> dict[str, float]:
    return {
        "eks": EKS_HOURLY * HOURS,
        "nodes": env.node_count * env.node_hourly * HOURS,
        "nat": env.nat_gateways * NAT_HOURLY * HOURS,
        "s3": env.s3_gb * S3_GB_MONTH,
        "gp3": env.gp3_gb * GP3_GB_MONTH,
    }


def overlaps(a: str, b: str) -> bool:
    return ipaddress.ip_network(a).overlaps(ipaddress.ip_network(b))


def main() -> None:
    for env in ENVIRONMENTS:
        blocks = subnets(env.vpc_cidr, 20)
        private = blocks[:3]
        public = blocks[8:11]
        print(f"{env.name:8} vpc={env.vpc_cidr:15} subnets={len(blocks):2d} "
              f"private={private}")
        print(f"{'':8} public={public}")

    print()
    for i, a in enumerate(ENVIRONMENTS):
        for b in ENVIRONMENTS[i + 1:]:
            state = "OVERLAP" if overlaps(a.vpc_cidr, b.vpc_cidr) else "ok"
            print(f"  {a.name:8} vs {b.name:8} {state}")

    print()
    total = 0.0
    for env in ENVIRONMENTS:
        parts = monthly_cost(env)
        subtotal = sum(parts.values())
        total += subtotal
        detail = " ".join(f"{k}={v:7.2f}" for k, v in parts.items())
        print(f"{env.name:8} {detail}  total={subtotal:8.2f} USD/month")
    print(f"{'all':8} {'':54} total={total:8.2f} USD/month")

    print()
    prod = ENVIRONMENTS[-1]
    single = dict(monthly_cost(prod))
    single["nat"] = 1 * NAT_HOURLY * HOURS
    print(f"prod with one NAT gateway: {sum(single.values()):.2f} USD/month")
    print(f"three-AZ NAT premium:      "
          f"{sum(monthly_cost(prod).values()) - sum(single.values()):.2f} USD/month")

    print()
    report = {"environments": [
        {"name": e.name, "vpc": e.vpc_cidr,
         "monthly_usd": round(sum(monthly_cost(e).values()), 2)}
        for e in ENVIRONMENTS
    ]}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

Actual output:

```text
dev      vpc=10.42.0.0/16    subnets=16 private=['10.42.0.0/20', '10.42.16.0/20', '10.42.32.0/20']
         public=['10.42.128.0/20', '10.42.144.0/20', '10.42.160.0/20']
staging  vpc=10.43.0.0/16    subnets=16 private=['10.43.0.0/20', '10.43.16.0/20', '10.43.32.0/20']
         public=['10.43.128.0/20', '10.43.144.0/20', '10.43.160.0/20']
prod     vpc=10.44.0.0/16    subnets=16 private=['10.44.0.0/20', '10.44.16.0/20', '10.44.32.0/20']
         public=['10.44.128.0/20', '10.44.144.0/20', '10.44.160.0/20']

  dev      vs staging  ok
  dev      vs prod     ok
  staging  vs prod     ok

dev      eks=  73.00 nodes=  60.74 nat=  32.85 s3=   2.30 gp3=   8.00  total=  176.89 USD/month
staging  eks=  73.00 nodes=  60.74 nat=  32.85 s3=   4.60 gp3=  16.00  total=  187.19 USD/month
prod     eks=  73.00 nodes= 714.82 nat=  98.55 s3=  11.50 gp3=  48.00  total=  945.87 USD/month
all                                                             total= 1309.94 USD/month

prod with one NAT gateway: 880.17 USD/month
three-AZ NAT premium:      65.70 USD/month

{
  "environments": [
    {
      "name": "dev",
      "vpc": "10.42.0.0/16",
      "monthly_usd": 176.89
    },
    {
      "name": "staging",
      "vpc": "10.43.0.0/16",
      "monthly_usd": 187.19
    },
    {
      "name": "prod",
      "vpc": "10.44.0.0/16",
      "monthly_usd": 945.87
    }
  ]
}
```

The script uses `ipaddress` rather than `cidrsubnet`, but the arithmetic is the same: extending the
prefix by four bits creates sixteen subnets and the indices select which ones. A useful extension
is to fail the script when any pair overlaps, which turns the check into a continuous integration
gate.

The following backend configuration is HCL. It is validated by parsing with `terraform fmt` and
`terraform validate` in a project with the AWS provider installed; it has not been applied.

```hcl
terraform {
  required_version = ">= 1.11.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.59"
    }
  }

  backend "s3" {
    bucket       = "biomed-tfstate-123456789012"
    key          = "prod/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}
```

The `>= 1.11.0` constraint is deliberate: `use_lockfile` is generally available from that release,
and an older Terraform would ignore it or reject it. For a regulated environment, pin an exact
version and upgrade on a schedule.

## 8. Data, standards, and regulatory context

### 8.1 Configuration management

IEC 62304 clause 8 requires configuration identification, control, and status accounting for
medical device software. An IaC repository satisfies the mechanics: every environment is a versioned
configuration item, every change is a reviewed commit, and the applied commit identifies the
environment. What it does not do by itself is prove that the running environment matches the commit.
That requires the drift check in Section 5, Step 11, and the evidence that it ran.

### 8.2 HIPAA Security Rule

The Security Rule does not require IaC, and no clause mentions Terraform. It does constrain the
outcomes that IaC makes auditable:

| Clause | Constraint | IaC contribution |
|---|---|---|
| 164.312(a)(1) Access control | Limit access to electronic PHI | Bucket policies, IAM roles, and security groups as reviewed code |
| 164.312(b) Audit controls | Record access and activity | CloudTrail plus the repository history of intended changes |
| 164.312(c)(1) Integrity | Protect against improper alteration | Versioning, public access block, immutable tags as code |
| 164.312(e)(1) Transmission security | Protect PHI in transit | TLS-only bucket policies and private endpoints as code |
| 164.308(a)(8) Evaluation | Periodic technical evaluation | Scheduled `terraform plan` as a drift report |

### 8.3 State as a regulated record

State contains the attribute values of every managed resource, including generated passwords and
keys. Treating it as a build artefact rather than a regulated record is a common mistake. The
controls are: encryption at rest, restricted read access, versioning for recovery, and a retention
period that matches the change-control policy. A state bucket with public access is a breach, not
a misconfiguration.

### 8.4 NIST and supply chain

NIST SP 800-218, the Secure Software Development Framework, includes practices for protecting the
build environment and for producing evidence of the build. An IaC pipeline with pinned providers,
a locked state, and a reviewed plan is aligned with those practices. It does not by itself satisfy
them; the pipeline must also protect credentials and record who approved what.

### 8.5 What the standards do not settle

They do not prescribe Terraform, a module structure, a state backend, or an environment count. They
constrain the evidence and the controls. A team using CloudFormation or Pulumi can satisfy every
clause above. The engineering choices remain yours.

## 9. Industry practice and case evidence

### 9.1 Documented practice

**S3-native locking is the current recommendation.** HashiCorp's S3 backend documentation describes
`use_lockfile` as the locking mechanism and marks the DynamoDB arguments as deprecated. The
Terraform 1.11.0 changelog, released 27 February 2025, states that S3 native state locking is
generally available and that the DynamoDB arguments are deprecated. AWS Prescriptive Guidance
recommends the same approach and calls the DynamoDB table a legacy approach.

**Module versioning is explicit.** The Terraform Registry lists `terraform-aws-modules/eks/aws`
21.29.0 as the current release, published 7 October 2026, requiring Terraform 1.5.7 or later and AWS
provider 6.59 or later. The module defines 82 resources and has migration guides for each major
version from 17 to 21.

**Managed node groups do not follow the control plane.** The EKS documentation states that managed
node groups are not automatically upgraded when the control plane is upgraded, and recommends
keeping the versions aligned. This is a Terraform behaviour as much as an EKS behaviour: the
`cluster_version` variable changes one resource, and the node group has its own version argument.

### 9.2 Case evidence

Public incident reports specific to Terraform in healthcare are scarce. The instructive public
record is about state and blast radius:

1. **The 2019 Code Spaces outage** is not a Terraform case, but it is the canonical example of what
   happens when an attacker with console access can delete backups and infrastructure. The lesson
   transferred to IaC is that the state bucket and the credentials that can destroy it are part of
   the blast radius.
2. **Multiple public post-mortems describe S3 state bucket deletion or versioning being disabled**
   as the cause of multi-day recovery efforts. The pattern is consistent: versioning was off, the
   bucket was deleted, and the infrastructure had to be imported resource by resource.
3. **The Terraform 0.12 upgrade** required configuration changes across the ecosystem and produced
   plans with unexpected replacements for teams that had not pinned providers. It is the reason
   version constraints are now standard practice.

### 9.3 Inference

The following are inferences, labelled as such:

- Teams that run one state per environment recover from mistakes faster than teams that run one
  state for everything, because the blast radius of a bad apply is bounded.
- Most "Terraform is too slow" complaints are about a monolithic state with hundreds of resources,
  not about the tool. Splitting by lifecycle reduces plan time and risk together.
- The most common production incident caused by IaC is an unintended replacement, not a failed
  apply. A failed apply is visible; a replacement that succeeds is a data-loss event that looks like
  a successful deployment.

A practical corollary follows. Teams that gate on `forces replacement` in review catch more
incidents than teams that gate on test coverage, because the failure is in the plan rather than in
the code. The gate is cheap to add: a script that parses `terraform show -json tfplan` and fails if
any stateful resource type appears with a replacement action. The list of stateful types is short:
buckets, databases, key stores, file systems, and anything with `prevent_destroy` set.

The gate has one false-positive class worth planning for: a resource that is replaced on purpose,
such as a launch template during a node group upgrade. Allow an explicit override with a written
reason, and require the reason to name the migration procedure. An override that says "planned" is
not a reason.

The second corollary concerns state growth. A module that creates one resource per tenant produces a
state file that grows with the customer base, and a plan that slows down proportionally. Splitting
by tenant or by lifecycle stage keeps both the plan time and the blast radius bounded. The decision
belongs with the module design, not with an operations ticket after the state reaches a hundred
megabytes.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Local state only | State file in a repository or laptop | No remote backend | Loss makes the environment unmanageable | Remote backend with versioning |
| No locking | Two applies overlap | Concurrent writers | Corrupted state | `use_lockfile` or equivalent |
| Unencrypted state | Bucket without SSE | Attribute values readable | Secret exposure | SSE-KMS on the state bucket |
| Unpinned provider | Plan changes without a commit | Provider upgrade | Unexpected replacements | Version constraints |
| Unpinned module | Module update changes defaults | Silent behaviour change | Drift and surprise | Pin major, upgrade deliberately |
| No plan review | Apply run directly from a laptop | Consequences unseen | Data loss | Saved plan and approval |
| Stale plan applied | Long gap between plan and apply | State changed | Apply conflicts or wrong result | Apply promptly, re-plan if delayed |
| Bucket name from random suffix | Lost state creates a new bucket | Random value regenerated | Apparent data loss | Restore state, import existing |
| `single_nat_gateway` in production | One NAT for all zones | Single point of failure | Outbound outage on zone loss | One NAT per zone |
| Public cluster endpoint | API reachable from internet | Convenience default | Brute force and credential attacks | Private endpoint or CIDR allowlist |
| No deletion protection | Destroy removes data | No lifecycle guard | Permanent loss | `prevent_destroy` on stateful resources |
| Tags missing | Cannot attribute cost | Untagged resources | Budget blind spots | Tag at module level, enforce with policy |
| Overlapping CIDRs | Peering fails | Unplanned address allocation | No hybrid connectivity | Allocate from a plan, check in CI |
| Credential expiry mid-apply | Apply fails after partial creation | Short-lived role session | Half-built environment | Longer session or smaller change sets |
| Plan applied out of order | Resources created before dependencies | Missing references | Apply errors, orphaned resources | Express dependencies through references |
| Provider state migration skipped | Schema warnings after upgrade | Major provider change | Plan errors or silent drift | Read the upgrade guide, test in staging |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| One state per environment | Blast-radius reasoning | Shared network resources | Inspect backend keys |
| Three availability zones exist | Subnet and NAT arithmetic | Region with two zones | `aws ec2 describe-availability-zones` |
| Node count is fixed at six | Cost arithmetic | Autoscaling changes the count | Read the node group's current size |
| Prices are on-demand | Cost comparison | Savings plans, spot | Recalculate with commitments |
| The state bucket exists and is versioned | Backend operation | Bootstrap skipped | `aws s3api get-bucket-versioning` |
| Provider constraints are compatible | Plan succeeds | Major version skew | `terraform init` output |
| Modules are the only source of resources | Drift reasoning | Console changes | Scheduled `terraform plan` |
| The apply identity has minimum permissions | Security claim | Broad administrator role | Review the role policy |
| State contains no long-lived secrets | Exposure claim | Generated passwords, keys | Scan state for sensitive attributes |
| The team reviews plans | Change control | Rubber-stamping large plans | Audit review records |

## 12. Exercises

### 12.1 Design the backend

**Deliverable.** A backend configuration for three environments with encryption, versioning, and
locking, plus a paragraph on how the state bucket is bootstrapped.

**Acceptance criteria.** Separate keys per environment. `use_lockfile` enabled. Bootstrap described
as a separate root or manual step. No credentials in the file.

**Time.** 30 minutes.

### 12.2 Pin versions

**Deliverable.** A `versions.tf` for a stack using the AWS provider, the EKS module, and the VPC
module, with a justification for each constraint.

**Acceptance criteria.** Constraints are valid semver. The justification distinguishes reproducibility
from patch currency. States what happens on a major upgrade.

**Time.** 25 minutes.

### 12.3 Compute a subnet plan

**Deliverable.** For a /16 VPC split with a four-bit extension, list the first three private and
first three public subnets, and compute usable addresses per subnet.

**Acceptance criteria.** Correct CIDR arithmetic. States the five reserved addresses per AWS
subnet. Shows no overlap with a second environment at 10.43.0.0/16.

**Time.** 30 minutes.

### 12.4 Cost an environment

**Deliverable.** A monthly cost for an environment with one EKS cluster, four `m7g.xlarge` nodes,
two NAT gateways, 300 GB of S3, and 400 GB of gp3, at us-east-2 in USD.

**Acceptance criteria.** Every price has a source and an access date. Arithmetic to the cent.
States the hours-per-month convention.

**Time.** 35 minutes.

### 12.5 Read a plan

**Deliverable.** Given a plan that changes a VPC CIDR, a list of every resource likely to be
replaced, in dependency order, and a recommendation.

**Acceptance criteria.** Identifies the VPC, subnets, route tables, NAT gateways, node group, and
cluster as affected. Recommends a new VPC rather than an in-place change.

**Time.** 40 minutes.

### 12.6 Diagnose a lock

**Deliverable.** A procedure for a state lock error, including how to determine whether the lock is
stale and the command to release it.

**Acceptance criteria.** Does not recommend force-unlock as a first step. Names the risk of
force-unlocking a live apply.

**Time.** 20 minutes.

### 12.7 Design promotion

**Deliverable.** A promotion path from development to staging to production using the same modules,
with separate state and a reviewed diff at each step.

**Acceptance criteria.** No state copying. Each environment has its own variables. The promotion
artefact is a commit, not a manual change.

**Time.** 50 minutes.

### 12.8 Protect state

**Deliverable.** A threat model for the state bucket covering read access, write access, deletion,
and version recovery, with one control per threat.

**Acceptance criteria.** Covers encryption, least privilege, versioning, MFA delete or equivalent,
and access logging. Names what an attacker with state read access can learn.

**Time.** 45 minutes.

## 13. Self-check answers

### 13.1 Backend design

A correct answer uses one bucket with keys `dev/terraform.tfstate`, `staging/terraform.tfstate`, and
`prod/terraform.tfstate`, `encrypt = true`, and `use_lockfile = true`. The bucket is created by a
separate bootstrap root with versioning and public access blocked, because the main configuration
cannot store its state in a bucket it has not created yet.

### 13.2 Version constraints

`required_version = ">= 1.11.0"` is needed for `use_lockfile`. The AWS provider might be pinned at
`~> 6.59` to match the EKS module's requirement. The EKS module at `~> 21.0` allows minor updates
within 21 and blocks 22. A major upgrade requires reading the migration guide and reviewing the
plan for replacements.

### 13.3 Subnet arithmetic

With a /16 and a four-bit extension, the prefix becomes /20. The first private subnets are
10.42.0.0/20, 10.42.16.0/20, and 10.42.32.0/20. The public subnets at indices 8 to 10 are
10.42.128.0/20, 10.42.144.0/20, and 10.42.160.0/20. Each /20 has 4,096 addresses and 4,091 usable in
AWS.

### 13.4 Cost

```text
EKS    1 × 0.10   × 730 =  73.00
Nodes  4 × 0.1632 × 730 = 476.54
NAT    2 × 0.045  × 730 =  65.70
S3     300 × 0.023      =   6.90
gp3    400 × 0.08       =  32.00
                          ------
total                     654.14 USD/month
```

### 13.5 Plan reading

A VPC CIDR change forces replacement of the VPC. Subnets reference the VPC, so they are replaced.
Route tables, NAT gateways, and the internet gateway depend on the VPC and are replaced. The EKS
cluster references the subnets, so it is replaced, and the node group with it. The recommendation is
a new VPC with a new CIDR and a migration plan, not an in-place change. The plan should be read to
confirm the dependency chain is modelled; if it is not, the apply may fail partway.

### 13.6 Lock procedure

Read the error for the lock identifier, the principal, and the timestamp. Check whether that
principal has a running pipeline. If it does, wait for it. If it does not, and the timestamp is old,
`terraform force-unlock <lock-id>`. The risk is that a live apply is still writing; force-unlocking
then allows two writers and can corrupt state.

### 13.7 Promotion design

A correct design has three root directories or three backend keys, each calling the same module
versions. Promotion is a commit that changes the variables for the next environment, applied after
a plan review. State is never copied between environments, because the resource identifiers differ.

### 13.8 State threat model

Read access exposes attribute values, including generated credentials. Write access allows an
attacker to remove resources from state, causing Terraform to create duplicates or orphan
infrastructure. Delete access combined with no versioning makes recovery impossible. Controls:
SSE-KMS, least-privilege bucket policy, versioning, access logging, and a separate break-glass role
for recovery.

## 14. Interview questions

**1. Why is Terraform state sensitive?**
A strong answer says it contains attribute values including generated credentials, and that read
access is equivalent to reading secrets. A weak answer says it is important.

**2. What does `use_lockfile` do?**
A strong answer explains S3-native locking through a `.tflock` object, its general availability in
Terraform 1.11.0, and the deprecation of the DynamoDB table. A weak answer says it locks state.

**3. Why pin a module version?**
A strong answer distinguishes reproducibility from currency and explains that a module update can
change defaults. A weak answer says "best practice".

**4. What does `forces replacement` mean?**
A strong answer explains that the resource cannot be updated in place and will be destroyed and
recreated, with data loss for stateful resources. A weak answer says the resource is recreated.

**5. How do you detect drift?**
A strong answer describes a scheduled plan and a review of its output, and notes that some drift is
expected. A weak answer says "run plan".

**6. Why is one NAT gateway per environment a risk?**
A strong answer explains the single point of failure for outbound traffic and quantifies the
three-zone premium. A weak answer says it is cheaper.

**7. What happens if you rename a resource address without a `moved` block?**
A strong answer says Terraform destroys the old object and creates a new one. A weak answer says
nothing happens.

**8. How do you bootstrap the state bucket?**
A strong answer describes a separate root or a manual step with versioning and public access
blocked, because the main configuration cannot depend on state it has not created. A weak answer
says "create it with Terraform".

**9. What is the blast radius of an apply?**
A strong answer ties it to the state boundary and the dependency graph. A weak answer says "the
environment".

**10. How would you promote a change from staging to production?**
A strong answer describes the same module versions, a separate state, a reviewed plan, and a
commit-based promotion. A weak answer says "copy the state".

## 15. Key terms

- **Apply** — Execution of a plan.
- **Backend** — Where state is stored and locked.
- **Blast radius** — Scope affected by one apply.
- **CIDR** — Classless inter-domain routing notation for address ranges.
- **`cidrsubnet`** — Terraform function that extends a network prefix.
- **Data source** — Read-only lookup during plan.
- **Drift** — Difference between state and remote objects.
- **ECR** — Elastic Container Registry.
- **EKS** — Elastic Kubernetes Service.
- **IaC** — Infrastructure as code.
- **Import** — Adopting an existing object into state.
- **KMS** — Key Management Service.
- **Lock** — Mechanism preventing concurrent state writes.
- **Module** — Reusable group of resources.
- **`moved` block** — Address change without recreation.
- **NAT gateway** — Managed network address translation.
- **Plan** — Proposed changes.
- **Provider** — Plugin mapping resources to an API.
- **Refresh** — Reconciliation of state with remote objects.
- **Root module** — Configuration you run Terraform in.
- **State** — Terraform's record of managed objects.
- **Subnet** — Address range within a VPC.
- **Tag** — Key-value metadata.
- **Taint** — Marking a resource for replacement.
- **VPC** — Virtual private cloud.
- **Workspace** — Named state within one configuration.

## 16. Sources

All sources accessed 26 October 2026.

1. HashiCorp. *Backend Type: s3*.
   https://developer.hashicorp.com/terraform/language/backend/s3
2. HashiCorp. *Terraform v1.11.0 changelog*.
   https://github.com/hashicorp/terraform/blob/v1.11.0/CHANGELOG.md
3. AWS Prescriptive Guidance. *Backend best practices*.
   https://docs.aws.amazon.com/prescriptive-guidance/latest/terraform-aws-provider-best-practices/backend.html
4. Terraform Registry. *terraform-aws-modules/eks/aws*.
   https://registry.terraform.io/modules/terraform-aws-modules/eks/aws/latest
5. Terraform Registry. *terraform-aws-modules/vpc/aws*.
   https://registry.terraform.io/modules/terraform-aws-modules/vpc/aws/latest
6. AWS. *Amazon EKS — Kubernetes version lifecycle*.
   https://docs.aws.amazon.com/eks/latest/userguide/kubernetes-versions.html
7. AWS. *Amazon VPC pricing*. https://aws.amazon.com/vpc/pricing/
8. AWS. *Amazon S3 pricing*. https://aws.amazon.com/s3/pricing/
9. AWS. *Amazon EBS pricing*. https://aws.amazon.com/ebs/pricing/
10. AWS. *EC2 On-Demand pricing*, US East (Ohio). https://aws.amazon.com/ec2/pricing/on-demand/
11. HHS. *HIPAA Security Rule*.
    https://www.hhs.gov/hipaa/for-professionals/security/index.html
12. NIST. *SP 800-218: Secure Software Development Framework*.
    https://csrc.nist.gov/pubs/sp/800/218/final
13. IEC. *IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes*.
    https://webstore.iec.ch/publication/22797

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_kubernetes.md`](03_kubernetes.md)
- Next topic: [`05_streaming.md`](05_streaming.md)
- Related topic: [`07_security.md`](07_security.md)
- Survey file: [`../../course/26_cloud_streaming_production_labs.md`](../../course/26_cloud_streaming_production_labs.md)
- Lab root: [`../../labs/`](../../labs/)
- Reference stack: [`../../labs/cloud/`](../../labs/cloud/)
- Terraform root: [`../../labs/cloud/terraform/main.tf`](../../labs/cloud/terraform/main.tf)
- Variables: [`../../labs/cloud/terraform/variables.tf`](../../labs/cloud/terraform/variables.tf)
- Backend example: [`../../labs/cloud/terraform/backend.tf.example`](../../labs/cloud/terraform/backend.tf.example)
- Network module: [`../../labs/cloud/terraform/modules/network/main.tf`](../../labs/cloud/terraform/modules/network/main.tf)
- Storage module: [`../../labs/cloud/terraform/modules/storage/main.tf`](../../labs/cloud/terraform/modules/storage/main.tf)
- EKS module: [`../../labs/cloud/terraform/modules/eks/main.tf`](../../labs/cloud/terraform/modules/eks/main.tf)
