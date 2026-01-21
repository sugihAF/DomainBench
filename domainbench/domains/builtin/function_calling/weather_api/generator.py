"""
Weather API function calling test case generator.

Generates test cases for all standard categories:
- simple: Single function call
- parallel: Multiple independent function calls
- multiple: Same function called multiple times
- multi_turn: Sequential conversation with state
- agentic: Complex tasks with text response validation
"""

import random
from typing import Any, Dict, List


# Data for generating test cases
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "London", "Paris", "Tokyo", "Sydney", "Toronto",
    "Berlin", "Madrid", "Rome", "Amsterdam", "Singapore",
    "Dubai", "Mumbai", "Seoul", "Bangkok", "Mexico City",
]

COUNTRIES = {
    "New York": "US", "Los Angeles": "US", "Chicago": "US",
    "Houston": "US", "Phoenix": "US", "London": "UK",
    "Paris": "FR", "Tokyo": "JP", "Sydney": "AU",
    "Toronto": "CA", "Berlin": "DE", "Madrid": "ES",
    "Rome": "IT", "Amsterdam": "NL", "Singapore": "SG",
    "Dubai": "AE", "Mumbai": "IN", "Seoul": "KR",
    "Bangkok": "TH", "Mexico City": "MX",
}

UNITS = ["celsius", "fahrenheit"]
SEVERITIES = ["low", "medium", "high", "critical"]
METRICS = ["temperature", "humidity", "wind_speed"]
REGIONS = ["California", "Texas", "Florida", "New York", "England", "Bavaria", "Ile-de-France"]

# Function definitions (OpenAI format)
FUNCTIONS = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather for a specified city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name to get weather for"},
                "country": {"type": "string", "description": "Country code (e.g., US, UK, JP)"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_forecast",
        "description": "Get weather forecast for upcoming days",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
                "days": {"type": "integer", "description": "Number of days to forecast (1-14)"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "days"],
        },
    },
    {
        "name": "get_weather_alerts",
        "description": "Get active weather alerts for a region",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Region or state name"},
                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            },
            "required": ["region"],
        },
    },
    {
        "name": "compare_weather",
        "description": "Compare weather between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "city1": {"type": "string", "description": "First city"},
                "city2": {"type": "string", "description": "Second city"},
                "metric": {"type": "string", "enum": ["temperature", "humidity", "wind_speed"]},
            },
            "required": ["city1", "city2"],
        },
    },
]

# Query templates
SIMPLE_TEMPLATES = [
    # get_current_weather
    ("What's the weather in {city}?", "get_current_weather(city='{city}')"),
    ("What's the current temperature in {city}?", "get_current_weather(city='{city}')"),
    ("Tell me the weather in {city} in {unit}", "get_current_weather(city='{city}', unit='{unit}')"),
    ("How's the weather in {city}, {country}?", "get_current_weather(city='{city}', country='{country}')"),
    # get_forecast
    ("What's the {days}-day forecast for {city}?", "get_forecast(city='{city}', days={days})"),
    ("Give me a {days} day weather forecast for {city}", "get_forecast(city='{city}', days={days})"),
    ("What will the weather be like in {city} for the next {days} days?", "get_forecast(city='{city}', days={days})"),
    # get_weather_alerts
    ("Are there any weather alerts in {region}?", "get_weather_alerts(region='{region}')"),
    ("Check for {severity} weather alerts in {region}", "get_weather_alerts(region='{region}', severity='{severity}')"),
    # compare_weather
    ("Compare the weather between {city1} and {city2}", "compare_weather(city1='{city1}', city2='{city2}')"),
    ("Which city is warmer, {city1} or {city2}?", "compare_weather(city1='{city1}', city2='{city2}', metric='temperature')"),
]

PARALLEL_TEMPLATES = [
    # Multiple weather lookups
    (
        "What's the weather in {city1} and {city2}?",
        ["get_current_weather(city='{city1}')", "get_current_weather(city='{city2}')"],
    ),
    (
        "Get me the weather for {city1}, {city2}, and {city3}",
        [
            "get_current_weather(city='{city1}')",
            "get_current_weather(city='{city2}')",
            "get_current_weather(city='{city3}')",
        ],
    ),
    (
        "What's the weather in {city1} and the 5-day forecast for {city2}?",
        ["get_current_weather(city='{city1}')", "get_forecast(city='{city2}', days=5)"],
    ),
    (
        "Check weather alerts in {region} and current weather in {city}",
        ["get_weather_alerts(region='{region}')", "get_current_weather(city='{city}')"],
    ),
]

MULTIPLE_TEMPLATES = [
    # Same function called multiple times in order
    (
        "Get the 3-day, 5-day, and 7-day forecasts for {city}",
        [
            "get_forecast(city='{city}', days=3)",
            "get_forecast(city='{city}', days=5)",
            "get_forecast(city='{city}', days=7)",
        ],
    ),
    (
        "Check weather in {city} in both celsius and fahrenheit",
        [
            "get_current_weather(city='{city}', unit='celsius')",
            "get_current_weather(city='{city}', unit='fahrenheit')",
        ],
    ),
]

# Multi-turn conversation templates
# Each template is a list of turns, where each turn has a query and expected calls
MULTI_TURN_TEMPLATES = [
    # Weather planning conversation
    {
        "scenario": "trip_planning",
        "turns": [
            {
                "query": "I'm planning a trip to {city}. What's the weather like there?",
                "expected_calls": ["get_current_weather(city='{city}')"],
            },
            {
                "query": "What about the forecast for the next {days} days?",
                "expected_calls": ["get_forecast(city='{city}', days={days})"],
            },
            {
                "query": "Are there any weather alerts I should know about in {region}?",
                "expected_calls": ["get_weather_alerts(region='{region}')"],
            },
        ],
    },
    # Comparison conversation
    {
        "scenario": "city_comparison",
        "turns": [
            {
                "query": "What's the current weather in {city1}?",
                "expected_calls": ["get_current_weather(city='{city1}')"],
            },
            {
                "query": "And how about {city2}?",
                "expected_calls": ["get_current_weather(city='{city2}')"],
            },
            {
                "query": "Can you compare the temperature between those two cities?",
                "expected_calls": ["compare_weather(city1='{city1}', city2='{city2}', metric='temperature')"],
            },
        ],
    },
    # Forecast deep dive
    {
        "scenario": "forecast_detail",
        "turns": [
            {
                "query": "What's the weather forecast for {city} for the next 3 days?",
                "expected_calls": ["get_forecast(city='{city}', days=3)"],
            },
            {
                "query": "Actually, can you extend that to 7 days?",
                "expected_calls": ["get_forecast(city='{city}', days=7)"],
            },
        ],
    },
    # Alert check conversation
    {
        "scenario": "alert_monitoring",
        "turns": [
            {
                "query": "Are there any critical weather alerts in {region}?",
                "expected_calls": ["get_weather_alerts(region='{region}', severity='critical')"],
            },
            {
                "query": "What about lower severity alerts?",
                "expected_calls": ["get_weather_alerts(region='{region}', severity='low')"],
            },
            {
                "query": "And what's the current weather in {city}?",
                "expected_calls": ["get_current_weather(city='{city}')"],
            },
        ],
    },
]


# Agentic templates - Complex reasoning tasks with text response validation
AGENTIC_TEMPLATES = [
    {
        "query": "Based on the weather data, should I bring an umbrella to {city} today? Just answer yes or no.",
        "context": "You have access to weather functions. Check the current weather and give a direct answer.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    },
    {
        "query": "I need to decide between visiting {city1} or {city2} this weekend. Which city has better weather? Just name the city.",
        "context": "Compare the weather between the two cities and recommend one.",
        "expected_response": ["{city1}", "{city2}"],
        "match_mode": "any",
    },
    {
        "query": "Is there any severe weather warning in {region}? Answer with 'yes, there is a warning' or 'no warnings'.",
        "context": "Check weather alerts for the region.",
        "expected_response": ["yes, there is a warning", "no warnings"],
        "match_mode": "any",
    },
    {
        "query": "What's the temperature trend for {city} over the next {days} days - warming up, cooling down, or staying stable?",
        "context": "Analyze the forecast data and describe the trend.",
        "expected_response": ["warming up", "cooling down", "staying stable"],
        "match_mode": "any",
    },
    {
        "query": "Should I pack winter clothes for my trip to {city}? Answer yes or no.",
        "context": "Check the weather and forecast to determine if winter clothes are needed.",
        "expected_response": ["yes", "no"],
        "match_mode": "any",
    },
]


# All supported categories
SUPPORTED_CATEGORIES = ["simple", "parallel", "multiple", "multi_turn", "agentic"]


def generate_test_cases(
    count: int,
    seed: int = 42,
    category: str = "simple",
) -> List[Dict[str, Any]]:
    """
    Generate function calling test cases.

    Args:
        count: Number of test cases to generate
        seed: Random seed for reproducibility
        category: Category to generate (simple, parallel, multiple, multi_turn, agentic, or all)

    Returns:
        List of test case dictionaries
    """
    rng = random.Random(seed)
    items = []

    if category == "all":
        # Mix of all categories
        categories = SUPPORTED_CATEGORIES.copy()
    elif category in SUPPORTED_CATEGORIES:
        categories = [category]
    else:
        raise ValueError(f"Unknown category: {category}. Supported: {', '.join(SUPPORTED_CATEGORIES)}, all")

    for i in range(count):
        cat = rng.choice(categories)

        if cat == "simple":
            item = _generate_simple(rng, i)
        elif cat == "parallel":
            item = _generate_parallel(rng, i)
        elif cat == "multiple":
            item = _generate_multiple(rng, i)
        elif cat == "multi_turn":
            item = _generate_multi_turn(rng, i)
        elif cat == "agentic":
            item = _generate_agentic(rng, i)

        items.append(item)

    return items


def _generate_simple(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a simple (single function call) test case."""
    template, gt_template = rng.choice(SIMPLE_TEMPLATES)

    city = rng.choice(CITIES)
    city1 = city
    city2 = rng.choice([c for c in CITIES if c != city])
    country = COUNTRIES.get(city, "US")
    unit = rng.choice(UNITS)
    days = rng.randint(1, 7)
    region = rng.choice(REGIONS)
    severity = rng.choice(SEVERITIES)

    query = template.format(
        city=city, city1=city1, city2=city2,
        country=country, unit=unit, days=days,
        region=region, severity=severity,
    )
    ground_truth = gt_template.format(
        city=city, city1=city1, city2=city2,
        country=country, unit=unit, days=days,
        region=region, severity=severity,
    )

    return {
        "id": f"weather_simple_{idx:04d}",
        "category": "simple",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }


def _generate_parallel(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a parallel (multiple independent calls) test case."""
    template, gt_templates = rng.choice(PARALLEL_TEMPLATES)

    cities = rng.sample(CITIES, 3)
    city = cities[0]
    city1, city2, city3 = cities
    region = rng.choice(REGIONS)

    query = template.format(
        city=city, city1=city1, city2=city2, city3=city3, region=region,
    )
    ground_truth = [
        gt.format(city=city, city1=city1, city2=city2, city3=city3, region=region)
        for gt in gt_templates
    ]

    return {
        "id": f"weather_parallel_{idx:04d}",
        "category": "parallel",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }


def _generate_multiple(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a multiple (same function called repeatedly) test case."""
    template, gt_templates = rng.choice(MULTIPLE_TEMPLATES)

    city = rng.choice(CITIES)

    query = template.format(city=city)
    ground_truth = [gt.format(city=city) for gt in gt_templates]

    return {
        "id": f"weather_multiple_{idx:04d}",
        "category": "multiple",
        "query": query,
        "functions": FUNCTIONS,
        "ground_truth": ground_truth,
    }


def _generate_multi_turn(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate a multi-turn conversation test case."""
    template = rng.choice(MULTI_TURN_TEMPLATES)

    # Generate random values for placeholders
    cities = rng.sample(CITIES, 2)
    city = cities[0]
    city1, city2 = cities
    region = rng.choice(REGIONS)
    days = rng.randint(3, 7)

    # Build turns with formatted values
    turns = []
    for turn_template in template["turns"]:
        query = turn_template["query"].format(
            city=city, city1=city1, city2=city2,
            region=region, days=days,
        )
        expected_calls = [
            call.format(city=city, city1=city1, city2=city2, region=region, days=days)
            for call in turn_template["expected_calls"]
        ]
        turns.append({
            "query": query,
            "expected_calls": expected_calls,
        })

    return {
        "id": f"weather_multi_turn_{idx:04d}",
        "category": "multi_turn",
        "query": turns[0]["query"],  # First turn query for compatibility
        "functions": FUNCTIONS,
        "turns": turns,
        "ground_truth": turns[0]["expected_calls"],  # First turn expected calls
    }


def _generate_agentic(rng: random.Random, idx: int) -> Dict[str, Any]:
    """Generate an agentic (text response) test case."""
    template = rng.choice(AGENTIC_TEMPLATES)

    # Generate random values for placeholders
    cities = rng.sample(CITIES, 2)
    city = cities[0]
    city1, city2 = cities
    region = rng.choice(REGIONS)
    days = rng.randint(3, 7)

    query = template["query"].format(
        city=city, city1=city1, city2=city2,
        region=region, days=days,
    )
    context = template.get("context", "").format(
        city=city, city1=city1, city2=city2,
        region=region, days=days,
    )

    # Format expected responses
    expected_response = template["expected_response"]
    if isinstance(expected_response, list):
        expected_response = [
            resp.format(city=city, city1=city1, city2=city2, region=region, days=days)
            for resp in expected_response
        ]
    else:
        expected_response = expected_response.format(
            city=city, city1=city1, city2=city2, region=region, days=days,
        )

    return {
        "id": f"weather_agentic_{idx:04d}",
        "category": "agentic",
        "query": query,
        "context": context,
        "functions": FUNCTIONS,
        "expected_response": expected_response,
        "match_mode": template.get("match_mode", "contains"),
    }
