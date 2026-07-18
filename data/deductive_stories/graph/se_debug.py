"""SE debugging domain: dependency graph → unique root-cause service."""

from __future__ import annotations

import random
from dataclasses import dataclass

from data.deductive_stories.graph.base import attach_gold, register_adapter
from data.deductive_stories.schema import (
    Entity,
    Event,
    EventGraph,
    Query,
    Relation,
)

SOLVER_ID = "se_debug_root_cause_v0"

TRAIN_TEMPLATES = (
    "se_debug.dep_cascade_v1",
    "se_debug.config_drift_v1",
)
TEST_TEMPLATES = (
    "se_debug.blast_radius_v1",
    "se_debug.symptom_backtrace_v1",
)

_SERVICES_TRAIN = (
    "auth-gateway", "user-service", "billing-api", "invoice-worker",
    "catalog-service", "search-indexer", "cache-proxy", "payment-ledger",
)
_SERVICES_TEST = (
    "edge-router", "session-store", "fulfillment-api", "inventory-sync",
    "notify-bus", "fraud-score", "tax-engine", "shipment-tracker",
)
_CONFIG_KEYS_TRAIN = (
    "DB_POOL_SIZE", "REDIS_TTL_SEC", "JWT_CLOCK_SKEW", "QUEUE_PREFETCH",
)
_CONFIG_KEYS_TEST = (
    "CIRCUIT_BREAKER_MS", "FEATURE_FLAG_TAX_V2", "GRPC_DEADLINE_MS", "CACHE_NS",
)


@dataclass
class SEDebugAdapter:
    domain: str = "se_debug"

    def generate(
        self,
        *,
        seed: int,
        split: str,
        distractor_ratio: float = 0.35,
        n_services: int = 6,
    ) -> EventGraph:
        rng = random.Random(seed)
        is_test = split in {"test", "test_ood_noise"}
        template_id = rng.choice(TEST_TEMPLATES if is_test else TRAIN_TEMPLATES)
        pool = list(_SERVICES_TEST if is_test else _SERVICES_TRAIN)
        config_keys = _CONFIG_KEYS_TEST if is_test else _CONFIG_KEYS_TRAIN
        rng.shuffle(pool)
        n_services = max(4, min(n_services, len(pool)))
        names = pool[:n_services]
        # Build a chain root → ... → leaf (symptom).
        root_idx = 0
        leaf_idx = n_services - 1
        config_key = rng.choice(config_keys)
        config_value = str(rng.choice([0, 1, 2, 8, 16, 32]))

        entities = [
            Entity(
                id=f"S{i}",
                type="service",
                name=name,
                attrs={
                    "is_root_cause": i == root_idx,
                    "is_symptom": i == leaf_idx,
                },
            )
            for i, name in enumerate(names)
        ]
        entities.append(
            Entity(
                id="C1",
                type="config",
                name=config_key,
                attrs={"value": config_value, "service_id": "S0"},
            )
        )

        events: list[Event] = []
        relations: list[Relation] = []
        # Dependency edges: S{i} depends_on S{i-1} (failure propagates root→leaf).
        for i in range(1, n_services):
            relations.append(
                Relation(src=f"S{i}", dst=f"S{i-1}", type="depends_on")
            )

        events.append(
            Event(
                id="V0",
                type="symptom",
                time=0,
                actors=[f"S{leaf_idx}"],
                attrs={},
                text_seed=(
                    f"On-call alert: {names[leaf_idx]} returned elevated 5xx rates."
                ),
            )
        )
        events.append(
            Event(
                id="V1",
                type="config_change",
                time=1,
                actors=["S0", "C1"],
                attrs={"key": config_key, "value": config_value},
                text_seed=(
                    f"Change log: {names[root_idx]} config {config_key} was set to "
                    f"{config_value} shortly before the incident."
                ),
            )
        )
        support = ["V0", "V1"]
        for i in range(1, n_services):
            eid = f"V{i+1}"
            events.append(
                Event(
                    id=eid,
                    type="propagation",
                    time=i + 1,
                    actors=[f"S{i}", f"S{i-1}"],
                    attrs={},
                    text_seed=(
                        f"{names[i]} depends on {names[i-1]}; errors cascaded when "
                        f"{names[i-1]} degraded."
                    ),
                )
            )
            support.append(eid)

        n_distractors = max(1, int(round(distractor_ratio * n_services)))
        for d in range(n_distractors):
            decoy = names[rng.randrange(1, n_services)]
            events.append(
                Event(
                    id=f"D{d}",
                    type="red_herring_log",
                    time=100 + d,
                    actors=[],
                    attrs={"distractor": True},
                    text_seed=(
                        f"Unrelated warning in {decoy}: disk usage crossed 70% on a "
                        f"canary host with no user impact."
                    ),
                )
            )

        blast_radius = n_services  # all services in the chain are affected
        queries = [
            Query(
                qid="Q1",
                type="who",
                prompt="Which service is the root cause of the incident?",
                answer_type="string_norm",
                support_node_ids=list(support),
                hop_depth=n_services - 1,
            ),
            Query(
                qid="Q2",
                type="what",
                prompt="Which configuration key was changed on the root-cause service?",
                answer_type="string_norm",
                support_node_ids=["V1"],
                hop_depth=1,
            ),
            Query(
                qid="Q3",
                type="number",
                prompt="How many services are in the dependency blast radius (including root)?",
                answer_type="number",
                support_node_ids=list(support),
                hop_depth=2,
            ),
        ]

        graph = EventGraph(
            domain="se_debug",
            template_id=template_id,
            seed=seed,
            entities=entities,
            events=events,
            relations=relations,
            queries=queries,
            gold=[],
            hidden_state={
                "root_service": names[root_idx],
                "config_key": config_key,
                "blast_radius": blast_radius,
                "symptom_service": names[leaf_idx],
            },
            distractor_ratio=float(distractor_ratio),
            noise_kind="in_domain_red_herring",
        )
        # Silence unused variable warning path by encoding blast in hidden_state.
        _ = blast_radius
        return attach_gold(graph, solver_name=SOLVER_ID)

    def solve(self, graph: EventGraph, query: Query) -> str:
        services = [e for e in graph.entities if e.type == "service"]
        roots = [e for e in services if e.attrs.get("is_root_cause")]
        if len(roots) != 1:
            raise ValueError(f"Expected one root cause, found {len(roots)}")
        configs = [e for e in graph.entities if e.type == "config"]
        if query.qid == "Q1":
            return roots[0].name
        if query.qid == "Q2":
            return configs[0].name
        if query.qid == "Q3":
            return str(len(services))
        raise KeyError(f"Unsupported se_debug query: {query.qid}")


register_adapter(SEDebugAdapter())
