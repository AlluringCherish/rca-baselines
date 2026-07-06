# RCA Baselines

이 저장소는 RCA baseline agent 코드, benchmark task metadata, 실험 결과, 그리고 압축된 telemetry 데이터를 포함합니다.

## Telemetry Data

원본 telemetry 디렉토리는 git에 직접 포함하지 않았습니다. 대신 dataset telemetry는 아래 경로에 분할 압축 파일로 들어 있습니다.

```text
RCA-datasets/telemetry_archive/
```

저장소를 clone한 뒤에는 반드시 repository root에서 압축을 풀어야 합니다. 압축 해제 위치가 중요합니다. 압축 파일은 `RCA-datasets/` 아래에 풀어야 하며, 최종 디렉토리 구조는 다음과 같아야 합니다.

```text
RCA-datasets/telemetry/
```

repository root에서 아래 명령을 실행하세요.

```bash
(cd RCA-datasets/telemetry_archive && sha256sum -c SHA256SUMS)
cat RCA-datasets/telemetry_archive/telemetry.tar.gz.part-* | sha256sum -c RCA-datasets/telemetry_archive/TELEMETRY_TAR_GZ_SHA256
cat RCA-datasets/telemetry_archive/telemetry.tar.gz.part-* > /tmp/telemetry.tar.gz
tar -xzf /tmp/telemetry.tar.gz -C RCA-datasets
```

repository root에 바로 압축을 풀면 `telemetry/`가 잘못된 위치에 생깁니다. benchmark 코드는 `RCA-datasets/telemetry/` 경로를 기준으로 동작합니다.
