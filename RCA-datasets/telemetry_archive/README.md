# Telemetry Archive

The raw `RCA-datasets/telemetry/` directory is excluded from git. It is stored here as split gzip archive chunks to stay below GitHub's 100 MB object limit.

Restore from the repository root:

```bash
(cd RCA-datasets/telemetry_archive && sha256sum -c SHA256SUMS)
cat RCA-datasets/telemetry_archive/telemetry.tar.gz.part-* | sha256sum -c RCA-datasets/telemetry_archive/TELEMETRY_TAR_GZ_SHA256
cat RCA-datasets/telemetry_archive/telemetry.tar.gz.part-* > /tmp/telemetry.tar.gz
tar -xzf /tmp/telemetry.tar.gz -C RCA-datasets
```

This recreates `RCA-datasets/telemetry/`.
