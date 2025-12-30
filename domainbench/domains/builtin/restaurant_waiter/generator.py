"""
Restaurant Waiter domain test case generator
Based on generate_waiterbench.py - generates multi-turn conversation scenarios
"""

import random
from typing import List, Dict, Any

# Categories for test scenarios
CATEGORIES = [
    "greeting_seating",
    "menu_qa",
    "allergy_safety",
    "dietary_constraints",
    "order_modifiers",
    "upsell_tactful",
    "ambiguity_clarify",
    "problem_wrong_order",
    "problem_quality_delay",
    "out_of_stock",
    "bill_split",
    "memory_across_turns",
]

# Data pools for generation
MEAL_TIMES = ["lunch", "dinner", "late-night"]
PARTY_SIZES = [1, 2, 3, 4, 5, 6]
DIETARY = ["vegetarian", "vegan", "pescatarian", "gluten-free", "dairy-free", "halal"]
ALLERGIES = ["peanuts", "tree nuts", "sesame", "shellfish", "eggs", "dairy", "gluten"]
SPICE_PREF = ["not spicy", "mild", "medium", "spicy", "extra spicy"]

ENTREES = [
    "grilled chicken bowl", "beef burger", "veggie burger", "margherita pizza",
    "pepperoni pizza", "pad thai", "fried rice", "ramen", "caesar salad",
    "greek salad", "tomato soup", "mushroom risotto", "fish tacos",
    "shrimp tacos", "steak sandwich", "tofu stir-fry", "salmon plate",
    "pasta alfredo", "pasta marinara", "chicken wings",
]

SIDES = ["fries", "side salad", "steamed veggies", "rice", "mashed potatoes", "fruit cup"]
DRINKS = ["water", "sparkling water", "iced tea", "lemonade", "cola", "coffee", "hot tea"]
DESSERTS = ["cheesecake", "chocolate brownie", "ice cream", "fruit sorbet", "apple pie"]

MODIFIERS = [
    "no onions", "no garlic", "sauce on the side", "extra sauce", "light sauce",
    "extra crispy", "no cheese", "add cheese", "gluten-free bun", "no mayo", "extra pickles",
]

COMPLAINTS = [
    "this is cold", "this is too salty", "this is overcooked", "this is undercooked",
    "this tastes different than usual", "we've been waiting a long time",
]

POLICY_FRAGMENTS = [
    "the kitchen closes in 20 minutes",
    "we're out of one ingredient",
    "we can't guarantee no cross-contamination",
    "we can remake it but it may take 10–15 minutes",
    "we can offer a refund or replacement",
]


def pick(rng: random.Random, xs: List) -> Any:
    return rng.choice(xs)


def picks(rng: random.Random, xs: List, k: int) -> List:
    return rng.sample(xs, k)


def maybe(rng: random.Random, p: float) -> bool:
    return rng.random() < p


def scenario_id(idx: int) -> str:
    return f"wb_{idx:04d}"


# Turn builders for each category
def build_turns_greeting_seating(rng: random.Random) -> List[str]:
    party = pick(rng, [2, 3, 4, 5, 6])
    time = pick(rng, MEAL_TIMES)
    t = [
        f"Hi! Table for {party}, please. It's {time}.",
        "How long is the wait? We're kind of in a hurry.",
        "Can we sit somewhere quieter if possible?",
    ]
    if maybe(rng, 0.4):
        t.append("Also, one person will arrive 10 minutes late—can we be seated now?")
    return t[: rng.randint(3, 4)]


def build_turns_menu_qa(rng: random.Random) -> List[str]:
    entree = pick(rng, ENTREES)
    spice = pick(rng, SPICE_PREF)
    t = [
        "What are your most popular dishes here?",
        f"Is the {entree} spicy? I prefer {spice}.",
        "What comes on it, and can you describe the portion size?",
    ]
    if maybe(rng, 0.5):
        t.append("What would you recommend if I want something lighter but still filling?")
    if maybe(rng, 0.3):
        t.append("Do you have anything kid-friendly?")
    return t[: rng.randint(3, 5)]


def build_turns_allergy_safety(rng: random.Random) -> List[str]:
    allergy = pick(rng, ALLERGIES)
    entree = pick(rng, ENTREES)
    t = [
        f"I have a severe allergy to {allergy}. What can I safely eat?",
        "Do you use shared fryers or have cross-contamination risks?",
        f"I was considering the {entree}. Is it safe or can it be made safe?",
    ]
    if maybe(rng, 0.6):
        t.append("If you can't guarantee safety, what are the safest options and why?")
    return t[: rng.randint(3, 4)]


def build_turns_dietary_constraints(rng: random.Random) -> List[str]:
    diet = pick(rng, DIETARY)
    entree = pick(rng, ENTREES)
    t = [
        f"I follow a {diet} diet. What do you recommend?",
        f"Does the {entree} fit that? If not, can you suggest an alternative?",
        "Can you confirm the ingredients and any hidden items like stock or sauce bases?",
    ]
    if maybe(rng, 0.5):
        t.append("Also, I'd like something with good protein—what's best?")
    return t[: rng.randint(3, 4)]


def build_turns_order_modifiers(rng: random.Random) -> List[str]:
    entree = pick(rng, ENTREES)
    m1, m2 = picks(rng, MODIFIERS, 2)
    side = pick(rng, SIDES)
    t = [
        f"I'd like the {entree}.",
        f"Can you do it with {m1} and {m2}?",
        f"Can I swap the default side for {side}?",
    ]
    if maybe(rng, 0.5):
        t.append("Actually, make that one of them on the side and the other extra.")
    if maybe(rng, 0.4):
        t.append("And can you confirm the final order back to me?")
    return t[: rng.randint(3, 5)]


def build_turns_upsell_tactful(rng: random.Random) -> List[str]:
    entree = pick(rng, ENTREES)
    drink = pick(rng, DRINKS)
    dessert = pick(rng, DESSERTS)
    t = [
        f"We'll start with two {entree}s.",
        "Do you have any specials or add-ons you recommend that go well with that?",
        f"What's a good drink pairing? Maybe something like {drink}?",
    ]
    if maybe(rng, 0.6):
        t.append(f"And if we get dessert, what's not too sweet—maybe {dessert}?")
    return t[: rng.randint(3, 4)]


def build_turns_ambiguity_clarify(rng: random.Random) -> List[str]:
    t = [
        "I'll get the pasta.",
        "Actually, make it the creamy one.",
        "I don't want it too heavy—what do you suggest?",
    ]
    if maybe(rng, 0.6):
        t.append("Also, can you make it mild but still flavorful?")
    if maybe(rng, 0.4):
        t.append("And I'm not sure if I want chicken or shrimp—what's better here?")
    return t[: rng.randint(3, 5)]


def build_turns_problem_wrong_order(rng: random.Random) -> List[str]:
    entree = pick(rng, ENTREES)
    t = [
        "Excuse me, this isn't what I ordered.",
        f"I ordered the {entree}, but this looks different.",
        "Can you fix it quickly? We're on a tight schedule.",
    ]
    if maybe(rng, 0.5):
        t.append("Can you also check on the rest of the table's food timing?")
    return t[: rng.randint(3, 4)]


def build_turns_problem_quality_delay(rng: random.Random) -> List[str]:
    complaint = pick(rng, COMPLAINTS)
    t = [
        f"Hi—{complaint}.",
        "Can you help resolve this?",
        "What are my options, and how long will it take?",
    ]
    if maybe(rng, 0.5):
        t.append("Can we get an update on the kitchen status too?")
    return t[: rng.randint(3, 4)]


def build_turns_out_of_stock(rng: random.Random) -> List[str]:
    entree = pick(rng, ENTREES)
    alt = pick(rng, [e for e in ENTREES if e != entree])
    diet = pick(rng, DIETARY)
    t = [
        f"I'd like the {entree}, please.",
        "Oh it's out of stock? What's the closest alternative?",
        f"Is the {alt} compatible with {diet}?",
    ]
    if maybe(rng, 0.5):
        t.append("If not, what would you recommend that is closest in flavor and price?")
    return t[: rng.randint(3, 4)]


def build_turns_bill_split(rng: random.Random) -> List[str]:
    n = pick(rng, [2, 3, 4])
    t = [
        f"Can we split the bill {n} ways?",
        "Put the appetizer on my card, and split the rest evenly.",
        "Can we get itemized receipts?",
    ]
    if maybe(rng, 0.4):
        t.append("Also, one person needs a separate receipt for a work expense—can you help?")
    return t[: rng.randint(3, 4)]


def build_turns_memory_across_turns(rng: random.Random) -> List[str]:
    allergy = pick(rng, ALLERGIES)
    spice_a = pick(rng, ["mild", "medium"])
    spice_b = pick(rng, ["spicy", "extra spicy"])
    entree1 = pick(rng, ENTREES)
    entree2 = pick(rng, [e for e in ENTREES if e != entree1])
    t = [
        f"Just a heads-up: someone at the table is allergic to {allergy}.",
        "We're deciding between a couple dishes—what's safe and what's popular?",
        f"Okay, we'll do {entree1} {spice_a} and {entree2} {spice_b}, but keep everything free of {allergy}.",
    ]
    if maybe(rng, 0.6):
        t.append("Can you repeat the full order back and confirm allergy safety/cross-contamination practices?")
    return t[: rng.randint(3, 4)]


# Map category to builder function
BUILDERS = {
    "greeting_seating": build_turns_greeting_seating,
    "menu_qa": build_turns_menu_qa,
    "allergy_safety": build_turns_allergy_safety,
    "dietary_constraints": build_turns_dietary_constraints,
    "order_modifiers": build_turns_order_modifiers,
    "upsell_tactful": build_turns_upsell_tactful,
    "ambiguity_clarify": build_turns_ambiguity_clarify,
    "problem_wrong_order": build_turns_problem_wrong_order,
    "problem_quality_delay": build_turns_problem_quality_delay,
    "out_of_stock": build_turns_out_of_stock,
    "bill_split": build_turns_bill_split,
    "memory_across_turns": build_turns_memory_across_turns,
}


def generate_test_cases(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate n test cases for the restaurant waiter domain.
    
    Args:
        n: Number of test cases to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of test case dictionaries
    """
    rng = random.Random(seed)
    items: List[Dict[str, Any]] = []

    # Ensure coverage: start with a round-robin over categories
    cat_cycle = []
    while len(cat_cycle) < n:
        cat_cycle.extend(CATEGORIES)
    cat_cycle = cat_cycle[:n]
    rng.shuffle(cat_cycle)

    for i in range(n):
        cat = cat_cycle[i]
        turns = BUILDERS[cat](rng)

        # Add occasional "policy pressure" turn to make it more realistic
        if maybe(rng, 0.25):
            turns = turns + [f"By the way, {pick(rng, POLICY_FRAGMENTS)}—what do you suggest we do?"]

        # Keep turns between 3 and 6
        if len(turns) < 3:
            turns += ["Can you confirm?"]
        turns = turns[: rng.randint(3, 6)]

        items.append({
            "id": scenario_id(i + 1),
            "category": cat,
            "turns": turns,
            "meta": {
                "meal_time": pick(rng, MEAL_TIMES),
                "party_size": pick(rng, PARTY_SIZES),
            },
        })

    return items


# Default static test cases (the original 12 from waiterbench.py)
DEFAULT_TEST_CASES = [
    {
        "id": "wb_001",
        "category": "greeting_seating",
        "turns": [
            "Hi! Table for two, please.",
            "We're in a hurry—what are the fastest items to order?",
            "Can we also get water right away?"
        ]
    },
    {
        "id": "wb_002",
        "category": "menu_qa",
        "turns": [
            "What's your most popular dish here?",
            "Is it spicy? What does it come with?",
            "What would you recommend for someone who doesn't eat beef?"
        ]
    },
    {
        "id": "wb_003",
        "category": "allergy_safety",
        "turns": [
            "I have a severe peanut allergy. What can I safely eat?",
            "Does your kitchen use peanut oil or share fryers?",
            "If it's not safe, what alternatives do you recommend?"
        ]
    },
    {
        "id": "wb_004",
        "category": "dietary_vegetarian",
        "turns": [
            "I'm vegetarian but I eat dairy. Any suggestions?",
            "Can you confirm the soup stock isn't meat-based?",
            "Great—can I get that, and a side salad with dressing on the side?"
        ]
    },
    {
        "id": "wb_005",
        "category": "order_modifiers",
        "turns": [
            "Can I order a burger medium-well with no onions?",
            "Add extra pickles, and sauce on the side.",
            "Also, can I swap fries for a salad?"
        ]
    },
    {
        "id": "wb_006",
        "category": "ambiguity_clarify",
        "turns": [
            "I'll get the pasta.",
            "Actually make it the creamy one.",
            "And can you make it not too heavy?"
        ]
    },
    {
        "id": "wb_007",
        "category": "upsell_tactful",
        "turns": [
            "We'll just have two entrees.",
            "Do you have any drink specials?",
            "What dessert would you recommend that's not too sweet?"
        ]
    },
    {
        "id": "wb_008",
        "category": "problem_wrong_order",
        "turns": [
            "Excuse me, this isn't what I ordered.",
            "I asked for no cheese—can you fix it?",
            "We're on a tight schedule—how long will it take?"
        ]
    },
    {
        "id": "wb_009",
        "category": "problem_cold_food",
        "turns": [
            "My food is cold.",
            "Can you reheat it or remake it?",
            "Also, could we get an update on the rest of the table's food?"
        ]
    },
    {
        "id": "wb_010",
        "category": "split_bill",
        "turns": [
            "Can we split the bill 3 ways?",
            "Put the appetizer on my card, and the rest evenly split.",
            "Can we also get the receipt emailed?"
        ]
    },
    {
        "id": "wb_011",
        "category": "out_of_stock",
        "turns": [
            "I'd like the salmon, please.",
            "Oh it's out of stock? What's the closest alternative?",
            "Okay—can you make sure it's gluten-free?"
        ]
    },
    {
        "id": "wb_012",
        "category": "memory_across_turns",
        "turns": [
            "We have a peanut allergy at the table, just so you know.",
            "We'll start with an appetizer to share—what's safe?",
            "Great. Now for mains: one spicy, one mild, and please keep everything peanut-free."
        ]
    },
]
