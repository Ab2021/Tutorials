# Lab 10.3: S3 Storage and Bucket Management

## Objective
Create and manage S3 buckets, upload/download files, and configure bucket policies.

## Prerequisites
- AWS account (Lab 10.1)
- AWS CLI configured

## Learning Objectives
- Create S3 buckets via Console and CLI
- Upload and manage objects
- Configure bucket permissions and policies
- Enable versioning and lifecycle rules
- Host static websites on S3

---

## Part 1: Create S3 Bucket (Console)

### Step 1: Navigate to S3

1. AWS Console → S3
2. Click "Create bucket"

### Step 2: Configure Bucket

**Bucket name:** `my-demo-bucket-<your-initials>-2024`  
(Must be globally unique!)

**Region:** us-east-1

**Object Ownership:** ACLs disabled

**Block Public Access:** Keep all checked (for now)

**Versioning:** Disabled

**Encryption:** SSE-S3 (default)

Click "Create bucket"

---

## Part 2: Upload and Manage Objects

### Upload File (Console)

1. Click on your bucket
2. Click "Upload"
3. Drag/drop or select files
4. Click "Upload"

### Upload via CLI

```bash
# Create test file
echo "Hello S3!" > test.txt

# Upload
aws s3 cp test.txt s3://my-demo-bucket-xyz-2024/

# List objects
aws s3 ls s3://my-demo-bucket-xyz-2024/

# Download
aws s3 cp s3://my-demo-bucket-xyz-2024/test.txt downloaded.txt
```

### Sync Directory

```bash
# Create local directory
mkdir website
echo "<h1>My Website</h1>" > website/index.html
echo "<p>About page</p>" > website/about.html

# Sync to S3
aws s3 sync website/ s3://my-demo-bucket-xyz-2024/website/

# Sync from S3
aws s3 sync s3://my-demo-bucket-xyz-2024/website/ local-copy/
```

---

## Part 3: Bucket Policies

### Make Bucket Public (for static website)

**Bucket Policy:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-demo-bucket-xyz-2024/*"
    }
  ]
}
```

Apply via CLI:

```bash
# Save policy to file
cat > bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::my-demo-bucket-xyz-2024/*"
    }
  ]
}
EOF

# Disable block public access
aws s3api put-public-access-block \
  --bucket my-demo-bucket-xyz-2024 \
  --public-access-block-configuration \
  "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

# Apply policy
aws s3api put-bucket-policy \
  --bucket my-demo-bucket-xyz-2024 \
  --policy file://bucket-policy.json
```

---

## Part 4: Static Website Hosting

### Enable Website Hosting

```bash
aws s3 website s3://my-demo-bucket-xyz-2024/ \
  --index-document index.html \
  --error-document error.html
```

### Upload Website Files

```bash
# Create simple website
cat > index.html << 'EOF'
<!DOCTYPE html>
<html>
<head><title>My S3 Website</title></head>
<body>
  <h1>Hello from S3!</h1>
  <p>This website is hosted on Amazon S3.</p>
</body>
</html>
EOF

# Upload
aws s3 cp index.html s3://my-demo-bucket-xyz-2024/ \
  --content-type "text/html"
```

### Access Website

URL format: `http://my-demo-bucket-xyz-2024.s3-website-us-east-1.amazonaws.com`

```bash
# Get website URL
aws s3api get-bucket-website \
  --bucket my-demo-bucket-xyz-2024
```

---

## Part 5: Versioning and Lifecycle

### Enable Versioning

```bash
aws s3api put-bucket-versioning \
  --bucket my-demo-bucket-xyz-2024 \
  --versioning-configuration Status=Enabled
```

### Test Versioning

```bash
# Upload file
echo "Version 1" > file.txt
aws s3 cp file.txt s3://my-demo-bucket-xyz-2024/

# Modify and upload again
echo "Version 2" > file.txt
aws s3 cp file.txt s3://my-demo-bucket-xyz-2024/

# List versions
aws s3api list-object-versions \
  --bucket my-demo-bucket-xyz-2024 \
  --prefix file.txt
```

### Lifecycle Policy

```json
{
  "Rules": [
    {
      "Id": "DeleteOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    },
    {
      "Id": "TransitionToIA",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 90,
          "StorageClass": "STANDARD_IA"
        }
      ]
    }
  ]
}
```

---

## Challenges

### Challenge 1: Cross-Region Replication

Set up replication to another region for disaster recovery.

### Challenge 2: S3 Event Notifications

Configure S3 to trigger a Lambda function when files are uploaded.

---

## Success Criteria

✅ Created S3 bucket  
✅ Uploaded and downloaded files  
✅ Configured bucket policy  
✅ Hosted static website on S3  
✅ Enabled versioning  

---

## Key Learnings

- **Bucket names are globally unique** - Choose carefully
- **S3 is object storage, not file system** - No directories, only prefixes
- **Versioning protects against accidental deletion** - But increases storage costs
- **S3 is cheap for storage** - $0.023/GB/month (Standard)

---

## Cleanup

```bash
# Delete all objects (including versions)
aws s3 rm s3://my-demo-bucket-xyz-2024/ --recursive

# Delete bucket
aws s3 rb s3://my-demo-bucket-xyz-2024/ --force
```

---

## Next Steps

- **Module 11:** Advanced Docker
- **Module 12:** Kubernetes Fundamentals

**Estimated Time:** 40 minutes  
**Difficulty:** Beginner
