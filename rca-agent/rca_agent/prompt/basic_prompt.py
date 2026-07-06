from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


BANK_CANDIDATES = """## POSSIBLE ROOT CAUSE REASONS:

- high CPU usage
- high memory usage
- network latency
- network packet loss
- high disk I/O read usage
- high disk space usage
- high JVM CPU load
- JVM Out of Memory (OOM) Heap

## POSSIBLE ROOT CAUSE COMPONENTS:

- apache01
- apache02
- Tomcat01
- Tomcat02
- Tomcat04
- Tomcat03
- MG01
- MG02
- IG01
- IG02
- Mysql01
- Mysql02
- Redis01
- Redis02"""


MARKET_CANDIDATES = """## POSSIBLE ROOT CAUSE COMPONENTS:

(if the root cause is at the node level, i.e., the root cause is a specific node)
- node-1
- node-2
- node-3
- node-4
- node-5
- node-6

(if the root cause is at the pod level, i.e., the root cause is a specific container)

- frontend-0
- frontend-1
- frontend-2
- frontend2-0
- shippingservice-0
- shippingservice-1
- shippingservice-2
- shippingservice2-0
- checkoutservice-0
- checkoutservice-1
- checkoutservice-2
- checkoutservice2-0
- currencyservice-0
- currencyservice-1
- currencyservice-2
- currencyservice2-0
- adservice-0
- adservice-1
- adservice-2
- adservice2-0
- emailservice-0
- emailservice-1
- emailservice-2
- emailservice2-0
- cartservice-0
- cartservice-1
- cartservice-2
- cartservice2-0
- productcatalogservice-0
- productcatalogservice-1
- productcatalogservice-2
- productcatalogservice2-0
- recommendationservice-0
- recommendationservice-1
- recommendationservice-2
- recommendationservice2-0
- paymentservice-0
- paymentservice-1
- paymentservice-2
- paymentservice2-0

(if the root cause is at the service level, i.e., if all pods of a specific service are faulty, the root cause is the service itself)

- frontend
- shippingservice
- checkoutservice
- currencyservice
- adservice
- emailservice
- cartservice
- productcatalogservice
- recommendationservice
- paymentservice

## POSSIBLE ROOT CAUSE REASONS:

- container CPU load
- container memory load
- container network packet retransmission
- container network packet corruption
- container network latency
- container packet loss
- container process termination
- container read I/O load
- container write I/O load
- node CPU load
- node CPU spike
- node memory consumption
- node disk read I/O consumption
- node disk write I/O consumption
- node disk space consumption"""


TELECOM_CANDIDATES = """## POSSIBLE ROOT CAUSE REASONS:

- CPU fault
- network delay
- network loss
- db connection limit
- db close

## POSSIBLE ROOT CAUSE COMPONENTS:

(if the root cause is at the node level, i.e., the root cause is a specific node)

- os_001
- os_002
- os_003
- os_004
- os_005
- os_006
- os_007
- os_008
- os_009
- os_010
- os_011
- os_012
- os_013
- os_014
- os_015
- os_016
- os_017
- os_018
- os_019
- os_020
- os_021
- os_022

(if the root cause is at the pod level, i.e., the root cause is a specific container)

- docker_001
- docker_002
- docker_003
- docker_004
- docker_005
- docker_006
- docker_007
- docker_008

(if the root cause is at the service level, i.e., if all pods of a specific service are faulty, the root cause is the service itself)

- db_001
- db_002
- db_003
- db_004
- db_005
- db_006
- db_007
- db_008
- db_009
- db_010
- db_011
- db_012
- db_013"""


HARDCODED_CANDIDATES = {
    "Bank": BANK_CANDIDATES,
    "Market/cloudbed-1": MARKET_CANDIDATES,
    "Market/cloudbed-2": MARKET_CANDIDATES,
    "Telecom": TELECOM_CANDIDATES,
}


RCAEVAL_SYSTEMS = {
    "RCAEval/re2-ob": "Online Boutique",
    "RCAEval/re2-ss": "Sock Shop",
    "RCAEval/re2-tt": "Train Ticket",
}


def _format_bullets(values: list[str]) -> str:
    return "\n".join(f"- {value}" for value in values)


def _raw_file_listing(task_dir: str) -> str:
    root = Path(task_dir)
    if not root.exists():
        return "- task directory is not present"
    files = []
    for item in sorted(root.glob("*.csv")):
        files.append(f"- {item.name}")
    return "\n".join(files) if files else "- no CSV files discovered"


def _directory_structure() -> str:
    return """## TELEMETRY DIRECTORY STRUCTURE

- You can access the telemetry files for the current task in CSV format.

- The telemetry files are placed directly in the current working directory for this task.

- Use the available CSV files directly by filename, e.g., `metric.csv` and `trace.csv`."""


def _schema_examples(task_dir: str) -> str:
    root = Path(task_dir)
    blocks = []
    if (root / "metric.csv").exists():
        blocks.append("""`metric.csv`

    ```csv
    timestamp,cmdb_id,kpi_name,value
    1614839400,IG01,JVM-Memory_7778_JVM_Memory_HeapMemoryUsage,21.2969
    ```""")
    if (root / "log.csv").exists():
        blocks.append("""`log.csv`

    ```csv
    timestamp,cmdb_id,log_name,value
    1614839400,Tomcat01,log.error_count,0.0
    ```""")
    if (root / "trace.csv").exists():
        blocks.append("""`trace.csv`

    ```csv
    timestamp,cmdb_id,trace_name,value
    1614839400,IG01,trace.self.duration_mean,0.361747
    ```""")
    if (root / "error_logs.csv").exists():
        blocks.append("""`error_logs.csv`

    ```csv
    timestamp,cmdb_id,message
    1614839400,Tomcat01,...
    ```""")
    return "\n\n".join(f"{idx}. {block}" for idx, block in enumerate(blocks, 1))


class BasicPrompt:
    def __init__(self, dataset_label: str, config: Dict[str, Any]) -> None:
        self.dataset_label = dataset_label
        self.config = config
        self.cand = self._build_candidates()

    def _build_candidates(self) -> str:
        if self.dataset_label.startswith("RCAEval/"):
            reasons = self.config.get("reasons", [])
            components = self.config.get("answer_candidates", [])
            return f"""## POSSIBLE ROOT CAUSE REASONS:

{_format_bullets(reasons)}

## POSSIBLE ROOT CAUSE COMPONENTS:

{_format_bullets(components)}"""
        return HARDCODED_CANDIDATES.get(self.dataset_label, BANK_CANDIDATES)

    def build_candidates(self, task_context: Dict[str, Any]) -> str:
        return self.cand

    def build_schema(self, task_context: Dict[str, Any]) -> str:
        task_dir = task_context.get("task_dir", "")
        raw_files = _raw_file_listing(task_dir)
        schema_examples = _schema_examples(task_dir)
        clarification = _clarification(self.dataset_label)

        return f"""{_directory_structure()}

Available CSV files for this task:

{raw_files}

## DATA SCHEMA

{schema_examples}

{self.cand}

{clarification}"""


def _clarification(dataset_label: str) -> str:
    if dataset_label == "Bank":
        return """## CLARIFICATION OF TELEMETRY DATA:

1. This microservice system is a banking platform.

2. The `metric.csv` file records normalized per-minute metric KPIs. The specific names of these KPIs can be found in the `kpi_name` field.

3. The `trace.csv` file records normalized per-minute trace features. The specific names of these features can be found in the `trace_name` field.

4. The `log.csv` file, when available, records normalized per-minute log features. The specific names of these features can be found in the `log_name` field.

5. In all normalized telemetry files, timestamp units are in seconds and floored to minute.

6. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""
    if dataset_label in {"Market/cloudbed-1", "Market/cloudbed-2"}:
        return """## CLARIFICATION OF TELEMETRY DATA:

1. This microservice system is an E-commerce platform which includes a failover mechanism, with each service deployed across multiple pods. Pod equals to Container in this system.

2. The `metric.csv` file records normalized per-minute metric KPIs. The specific names of these KPIs can be found in the `kpi_name` field.

3. The `trace.csv` file records normalized per-minute trace features. The specific names of these features can be found in the `trace_name` field.

4. The `log.csv` file, when available, records normalized per-minute log features. The specific names of these features can be found in the `log_name` field.

5. The `cmdb_id` is the name of specific components, including nodes, pods, services, etc.

6. In all normalized telemetry files, timestamp units are in seconds and floored to minute.

7. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""
    if dataset_label == "Telecom":
        return """## CLARIFICATION OF TELEMETRY DATA:

1. This service system is a telecom database system.

2. The `metric.csv` file records normalized per-minute metric KPIs. The specific names of these KPIs can be found in the `kpi_name` field.

3. The `trace.csv` file records normalized per-minute trace features. The specific names of these features can be found in the `trace_name` field.

4. In all normalized telemetry files, timestamp units are in seconds and floored to minute.

5. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""
    if dataset_label.startswith("RCAEval/"):
        system_name = RCAEVAL_SYSTEMS.get(dataset_label, "microservice")
        return f"""## CLARIFICATION OF TELEMETRY DATA:

1. This microservice system is the {system_name} system.

2. The `metric.csv` file records normalized per-minute metric KPIs. The specific names of these KPIs can be found in the `kpi_name` field.

3. The `trace.csv` file records normalized per-minute trace features when trace telemetry is available. The specific names of these features can be found in the `trace_name` field.

4. The `log.csv` file, when available, records normalized per-minute log features. The specific names of these features can be found in the `log_name` field.

5. In all normalized telemetry files, timestamp units are in seconds and floored to minute.

6. Please use UTC in all analysis steps because the incident instructions are expressed in UTC. For filtering, parse Unix timestamp columns with `pd.to_datetime(..., unit='s', utc=True)` and compare them against UTC incident windows. Do not reinterpret UTC incident windows as UTC+8 or local time.

7. The issue asks for the root-cause service/component and the failure reason."""
    return """## CLARIFICATION OF TELEMETRY DATA:

1. The telemetry files are normalized per-minute CSV files for the current task.

2. Please use the UTC+8 time zone in all analysis steps since system is deployed in China/Hong Kong/Singapore."""


def make_basic_prompt(dataset_label: str, config: Dict[str, Any]) -> BasicPrompt:
    return BasicPrompt(dataset_label, config)
