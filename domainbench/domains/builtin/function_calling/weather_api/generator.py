"""
Weather API function calling test case generator.

Generates test cases for simple, parallel, and multiple categories.
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
        category: Category to generate (simple, parallel, multiple, or all)

    Returns:
        List of test case dictionaries
    """
    rng = random.Random(seed)
    items = []

    if category == "all":
        # Mix of all categories
        categories = ["simple", "parallel", "multiple"]
    else:
        categories = [category]

    for i in range(count):
        cat = rng.choice(categories)

        if cat == "simple":
            item = _generate_simple(rng, i)
        elif cat == "parallel":
            item = _generate_parallel(rng, i)
        elif cat == "multiple":
            item = _generate_multiple(rng, i)
        else:
            item = _generate_simple(rng, i)

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
