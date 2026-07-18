"""Detective domain: means/motive/opportunity unique-culprit graphs."""

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

SOLVER_ID = "detective_unique_triple_v0"

# Disjoint template families for train vs test firewall.
TRAIN_TEMPLATES = (
    "detective.means_motive_v1",
    "detective.alibi_gap_v1",
)
TEST_TEMPLATES = (
    "detective.unique_intersection_v1",
    "detective.chain_of_custody_v1",
)

_FIRST = (
    "Avery", "Blake", "Casey", "Drew", "Ellis", "Finley", "Harper", "Jordan",
    "Quinn", "Riley", "Morgan", "Parker", "Reese", "Sawyer", "Taylor",
)
_LAST_TRAIN = (
    "Cole", "Hayes", "Brooks", "Reed", "Stone", "Walsh", "Frost", "Lane",
)
_LAST_TEST = (
    "Voss", "Kline", "North", "Perez", "Singh", "Okada", "Niemi", "Duval",
)
_WEAPONS_TRAIN = ("brass candlestick", "kitchen knife", "iron poker", "garrote")
_WEAPONS_TEST = ("antique pistol", "climbing rope", "chemist's vial", "letter opener")
_LOCATIONS_TRAIN = ("study", "conservatory", "library", "cellar")
_LOCATIONS_TEST = ("observatory", "boathouse", "gallery", "wine vault")


@dataclass
class DetectiveAdapter:
    domain: str = "detective"

    def generate(
        self,
        *,
        seed: int,
        split: str,
        distractor_ratio: float = 0.35,
        n_suspects: int = 5,
    ) -> EventGraph:
        rng = random.Random(seed)
        is_test = split in {"test", "test_ood_noise"}
        template_id = rng.choice(TEST_TEMPLATES if is_test else TRAIN_TEMPLATES)
        lasts = _LAST_TEST if is_test else _LAST_TRAIN
        weapons = _WEAPONS_TEST if is_test else _WEAPONS_TRAIN
        locations = _LOCATIONS_TEST if is_test else _LOCATIONS_TRAIN

        n_suspects = max(3, n_suspects)
        names = []
        used = set()
        while len(names) < n_suspects + 1:
            name = f"{rng.choice(_FIRST)} {rng.choice(lasts)}"
            if name not in used:
                used.add(name)
                names.append(name)
        victim_name, *suspect_names = names
        culprit_idx = rng.randrange(n_suspects)
        weapon = rng.choice(weapons)
        location = rng.choice(locations)

        entities = [
            Entity(id="E0", type="victim", name=victim_name, attrs={}),
        ]
        for i, name in enumerate(suspect_names):
            is_culprit = i == culprit_idx
            # Culprit has all three; others get a proper subset.
            if is_culprit:
                motive, opportunity, means = True, True, True
            else:
                flags = [True, True, False]
                rng.shuffle(flags)
                # Ensure not all three accidentally.
                if all(flags):
                    flags[rng.randrange(3)] = False
                motive, opportunity, means = flags
            entities.append(
                Entity(
                    id=f"E{i+1}",
                    type="suspect",
                    name=name,
                    attrs={
                        "motive": motive,
                        "opportunity": opportunity,
                        "means": means,
                        "is_culprit": is_culprit,
                    },
                )
            )

        entities.append(
            Entity(id="W1", type="weapon", name=weapon, attrs={})
        )
        entities.append(
            Entity(id="L1", type="location", name=location, attrs={})
        )

        events: list[Event] = []
        relations: list[Relation] = []
        t = 0
        events.append(
            Event(
                id="V0",
                type="crime_discovered",
                time=t,
                actors=["E0"],
                attrs={"location_id": "L1"},
                text_seed=f"{victim_name} was found dead in the {location}.",
            )
        )
        t += 1
        events.append(
            Event(
                id="V1",
                type="weapon_found",
                time=t,
                actors=["W1"],
                attrs={},
                text_seed=f"Investigators recovered a {weapon} near the scene.",
            )
        )
        t += 1

        support_for_culprit: list[str] = ["V0", "V1"]
        for ent in entities:
            if ent.type != "suspect":
                continue
            attrs = ent.attrs
            for key, label in (
                ("motive", "motive"),
                ("opportunity", "opportunity"),
                ("means", "access to the means"),
            ):
                eid = f"V{t}"
                present = bool(attrs[key])
                if present:
                    fact_seed = f"{ent.name} had {label}."
                else:
                    fact_seed = f"{ent.name} lacked {label}."
                events.append(
                    Event(
                        id=eid,
                        type=f"evidence_{key}",
                        time=t,
                        actors=[ent.id],
                        attrs={"present": present, "kind": key},
                        text_seed=fact_seed,
                    )
                )
                relations.append(Relation(src=eid, dst="V0", type="about_crime"))
                if attrs.get("is_culprit"):
                    support_for_culprit.append(eid)
                t += 1

        # In-domain red herrings (distractor events that do not change gold).
        n_distractors = max(1, int(round(distractor_ratio * n_suspects * 2)))
        for d in range(n_distractors):
            decoy = rng.choice([e for e in entities if e.type == "suspect"])
            eid = f"D{d}"
            events.append(
                Event(
                    id=eid,
                    type="red_herring",
                    time=t,
                    actors=[decoy.id],
                    attrs={"distractor": True},
                    text_seed=(
                        f"A neighbour reported seeing {decoy.name} arguing with a "
                        f"delivery driver earlier that week, unrelated to the crime."
                    ),
                )
            )
            t += 1

        culprit = entities[culprit_idx + 1]
        queries = [
            Query(
                qid="Q1",
                type="who",
                prompt=f"Who murdered {victim_name}?",
                answer_type="string_norm",
                support_node_ids=list(support_for_culprit),
                hop_depth=3,
            ),
            Query(
                qid="Q2",
                type="what",
                prompt="What weapon was used?",
                answer_type="string_norm",
                support_node_ids=["V1"],
                hop_depth=1,
            ),
            Query(
                qid="Q3",
                type="where",
                prompt="Where was the body discovered?",
                answer_type="string_norm",
                support_node_ids=["V0"],
                hop_depth=1,
            ),
        ]

        graph = EventGraph(
            domain="detective",
            template_id=template_id,
            seed=seed,
            entities=entities,
            events=events,
            relations=relations,
            queries=queries,
            gold=[],
            hidden_state={
                "culprit_id": culprit.id,
                "weapon": weapon,
                "location": location,
                "victim": victim_name,
            },
            distractor_ratio=float(distractor_ratio),
            noise_kind="in_domain_red_herring",
        )
        # Stash display answers in hidden_state; solve() returns names.
        return attach_gold(graph, solver_name=SOLVER_ID)

    def solve(self, graph: EventGraph, query: Query) -> str:
        suspects = [e for e in graph.entities if e.type == "suspect"]
        culprits = [e for e in suspects if e.attrs.get("motive") and e.attrs.get("opportunity") and e.attrs.get("means")]
        if len(culprits) != 1:
            raise ValueError(f"Expected unique culprit, found {len(culprits)}")
        culprit = culprits[0]
        weapons = [e for e in graph.entities if e.type == "weapon"]
        locations = [e for e in graph.entities if e.type == "location"]
        if query.qid == "Q1" or query.type == "who":
            return culprit.name
        if query.qid == "Q2" or query.type == "what":
            return weapons[0].name
        if query.qid == "Q3" or query.type == "where":
            return locations[0].name
        raise KeyError(f"Unsupported detective query: {query.qid}")


register_adapter(DetectiveAdapter())
