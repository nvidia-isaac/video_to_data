# Credentials

Everything in this directory except this README is ignored by git. Put your
secrets here; never commit them.

## Object-storage credentials

Pipelines that pull datasets from the S3-compatible GCS endpoint read a JSON
file from this directory, for example `credentials/gcp_training.secret`:

```json
{
  "aws_access_key_id": "...",
  "aws_secret_access_key": "...",
  "endpoint_url": "https://storage.googleapis.com",
  "region_name": "us-central1"
}
```

`endpoint_url` is required because the transfer uses the S3 protocol against a
GCS bucket, so the `s3://` URIs in configs are not AWS endpoints.

The file path is a parameter, not a hard-coded constant, so different workflows
may point at different files in this directory.

## Environment variable alternative

CI and container runs that cannot mount this directory may instead export the
same JSON, base64-encoded, in an environment variable. When that variable is
set it takes precedence over the file:

```bash
export COSMOS_GCP_CHECKPOINT_CREDS="$(base64 -w0 credentials/gcp_training.secret)"
```
