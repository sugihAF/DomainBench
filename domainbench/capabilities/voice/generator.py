"""
Voice scenario generator.

Two modes:
  1. AI-powered: User provides a domain description; an LLM generates a
     complete scenario (system prompt, knowledge base, tools, turns).
  2. Built-in template: Pre-defined hotel_concierge scenario for quick testing.
"""

import json
import random
from typing import List, Dict, Any, Optional

from domainbench.capabilities.voice.config import VoiceScenario, VoiceTurn
from domainbench.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# AI-powered generation
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = """\
You are an expert at creating evaluation scenarios for voice AI agents.

Given a domain description, you will create a complete multi-turn conversation \
scenario that tests an AI assistant's ability to:
1. Answer questions grounded in a knowledge base
2. Use tools/functions correctly
3. Follow multi-turn instructions
4. Deflect out-of-scope questions

## Output Format

Return STRICT JSON (no markdown fences):
{
  "system_prompt": "Role definition for the AI assistant...",
  "knowledge_base": "Detailed factual content (5000-15000 characters) the assistant should reference...",
  "tools": [
    {
      "name": "tool_name",
      "description": "What this tool does",
      "parameters": {
        "type": "object",
        "properties": {
          "param1": {"type": "string", "description": "..."},
          "param2": {"type": "integer", "description": "..."}
        },
        "required": ["param1"]
      }
    }
  ],
  "turns": [
    {
      "input": "User's spoken input",
      "golden_text": "Expected ideal response from the assistant",
      "required_function_call": null,
      "function_call_response": null
    },
    {
      "input": "User asks something requiring a tool call",
      "golden_text": "I'll do that for you right away.",
      "required_function_call": {
        "name": "tool_name",
        "args": {"param1": "value"}
      },
      "function_call_response": {"status": "success", "detail": "..."}
    }
  ]
}

## Requirements

1. Create 3-6 tools relevant to the domain.
2. Create 15-30 turns forming a coherent, realistic conversation.
3. At least 5 turns should require function calls.
4. Include 1-2 out-of-scope questions the assistant should deflect.
5. Include turns that test knowledge base grounding (specific facts).
6. The knowledge base should contain enough detail (dates, names, numbers, \
locations) to verify factual accuracy.
7. Function call arguments should be specific and verifiable.
8. Include an end_session tool that ends the conversation (last turn).
9. Make the conversation feel natural — not like a test script.
"""


def generate_with_ai(
    domain_name: str,
    domain_description: str,
    provider: BaseProvider,
    model: str,
    num_turns: int = 20,
    scenario_id: Optional[str] = None,
    max_retries: int = 2,
) -> VoiceScenario:
    """
    Generate a voice scenario using an LLM.

    Args:
        domain_name: Short name for the domain.
        domain_description: Detailed description of the voice agent's role.
        provider: LLM provider to use for generation.
        model: Model name for generation.
        num_turns: Target number of conversation turns.
        scenario_id: Optional ID; auto-generated if not provided.
        max_retries: Number of retries on parse failure.

    Returns:
        A complete VoiceScenario.
    """
    user_prompt = (
        f"Create a voice agent evaluation scenario for the following domain:\n\n"
        f"**Domain**: {domain_name}\n"
        f"**Description**: {domain_description}\n"
        f"**Target turns**: {num_turns}\n\n"
        f"The AI assistant should be helpful, professional, and domain-appropriate. "
        f"Include realistic edge cases and challenging multi-turn interactions."
    )

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(max_retries + 1):
        response = provider.chat_completion(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=16000,
        )

        text = response.get("content", "").strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n", 1)
            text = lines[1] if len(lines) > 1 else ""
            if text.endswith("```"):
                text = text[:-3].strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": text})
                messages.append({
                    "role": "user",
                    "content": "Output was not valid JSON. Return ONLY strict JSON.",
                })
                continue
            raise ValueError(
                f"Failed to parse generator output after {max_retries + 1} attempts."
            )

        sid = scenario_id or f"voice_{domain_name.lower().replace(' ', '_')}_001"
        turns = []
        for t in data.get("turns", []):
            fc = t.get("required_function_call")
            fr = t.get("function_call_response")
            turns.append(VoiceTurn(
                input=t.get("input", ""),
                golden_text=t.get("golden_text", ""),
                required_function_call=fc if fc else None,
                function_call_response=fr if fr else None,
            ))

        return VoiceScenario(
            id=sid,
            domain=domain_name,
            system_prompt=data.get("system_prompt", ""),
            knowledge_base=data.get("knowledge_base", ""),
            tools=data.get("tools", []),
            turns=turns,
        )

    raise ValueError("Unreachable: generation loop exhausted.")


# ---------------------------------------------------------------------------
# Built-in hotel concierge template
# ---------------------------------------------------------------------------

HOTEL_KNOWLEDGE_BASE = """\
# Grand Horizon Hotel — Guest Information

## General Information
- Address: 450 Pacific Avenue, San Francisco, CA 94133
- Phone: (415) 555-0199
- Check-in: 3:00 PM | Check-out: 11:00 AM
- Late check-out available until 2:00 PM ($50 fee)
- Free WiFi: Network "GrandHorizon-Guest", Password "welcome2025"
- Valet parking: $45/night | Self-park garage: $30/night

## On-Site Restaurants

### The Horizon Grill (Floor 1)
- Breakfast: 6:30 AM - 10:30 AM
- Lunch: 11:30 AM - 2:30 PM
- Dinner: 5:30 PM - 10:00 PM
- Cuisine: American contemporary
- Signature dish: Pan-seared salmon with citrus glaze ($38)
- Reservations recommended for dinner

### Sakura Lounge (Floor 12, Rooftop)
- Hours: 4:00 PM - 12:00 AM (closed Mondays)
- Cuisine: Japanese small plates and cocktails
- Happy hour: 4:00 PM - 6:00 PM (20% off all drinks)
- No reservations; first-come, first-served
- Dress code: Smart casual

### Cafe Mornings (Lobby)
- Hours: 5:30 AM - 3:00 PM daily
- Coffee, pastries, light sandwiches
- Grab-and-go available

## Room Service
- Available 24/7
- Menu covers breakfast, lunch, dinner, and late-night snacks
- Delivery fee: $8 per order
- Estimated delivery: 25-40 minutes

## Spa & Wellness
- Serenity Spa (Floor 3)
- Hours: 7:00 AM - 9:00 PM daily
- Services: Swedish massage ($120/60min), Deep tissue ($140/60min), \
Facial ($95/45min), Hot stone ($160/90min)
- Fitness center: 24/7 access with room key (Floor 3)
- Pool: Heated outdoor pool on Floor 4, open 6:00 AM - 10:00 PM

## Local Attractions
- Fisherman's Wharf: 0.8 miles (15-minute walk)
- Golden Gate Bridge: 3.2 miles (15 minutes by car)
- Chinatown: 0.3 miles (5-minute walk)
- Alcatraz Island: Ferry from Pier 33 (book 2+ weeks in advance)
- Cable car stop: 1 block east on Powell Street
- Union Square shopping: 0.6 miles south

## Transportation
- Airport shuttle (SFO): Runs every 60 minutes, $25/person, \
departs from lobby at :00 and :30
- Complimentary town car service within 3-mile radius (subject to availability)
- Uber/Lyft pickup: Designated area on Pacific Avenue side entrance
- BART station (Embarcadero): 0.9 miles, 18-minute walk

## Conference Facilities
- Grand Ballroom (Floor 2): Up to 300 guests
- Meeting rooms A-D (Floor 2): 10-50 guests each
- AV equipment included; catering available through The Horizon Grill
- Business center (Lobby level): Printing, scanning, fax — open 24/7

## Policies
- Pets: Dogs under 30 lbs allowed ($75/night pet fee)
- Smoking: Non-smoking property; designated outdoor area on Floor 1 patio
- Noise quiet hours: 10:00 PM - 7:00 AM
- Extra guest fee: $40/night for more than 2 guests per standard room
"""

HOTEL_TOOLS = [
    {
        "name": "book_restaurant",
        "description": "Book a restaurant reservation at an on-site restaurant.",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_name": {"type": "string", "description": "Guest's full name"},
                "restaurant": {"type": "string", "description": "Restaurant name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM format"},
                "party_size": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["guest_name", "restaurant", "date", "time", "party_size"],
        },
    },
    {
        "name": "book_spa",
        "description": "Book a spa appointment.",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_name": {"type": "string", "description": "Guest's full name"},
                "service": {"type": "string", "description": "Spa service name"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM format"},
            },
            "required": ["guest_name", "service", "date", "time"],
        },
    },
    {
        "name": "request_room_service",
        "description": "Place a room service order.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_number": {"type": "string", "description": "Guest's room number"},
                "items": {"type": "string", "description": "Comma-separated list of items"},
                "special_instructions": {"type": "string", "description": "Any special requests"},
            },
            "required": ["room_number", "items"],
        },
    },
    {
        "name": "schedule_transportation",
        "description": "Schedule airport shuttle or town car.",
        "parameters": {
            "type": "object",
            "properties": {
                "guest_name": {"type": "string", "description": "Guest's full name"},
                "service_type": {"type": "string", "description": "shuttle or town_car"},
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Pickup time in HH:MM format"},
                "destination": {"type": "string", "description": "Destination address or name"},
            },
            "required": ["guest_name", "service_type", "date", "time", "destination"],
        },
    },
    {
        "name": "submit_maintenance_request",
        "description": "Submit a room maintenance or housekeeping request.",
        "parameters": {
            "type": "object",
            "properties": {
                "room_number": {"type": "string", "description": "Guest's room number"},
                "issue_description": {"type": "string", "description": "Description of the issue"},
                "urgency": {"type": "string", "description": "low, medium, or high"},
            },
            "required": ["room_number", "issue_description"],
        },
    },
    {
        "name": "end_session",
        "description": "End the conversation when the guest is done.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]

HOTEL_SYSTEM_PROMPT = """\
You are a friendly and professional voice concierge for the Grand Horizon Hotel \
in San Francisco. Your job is to assist hotel guests with questions about the \
hotel, local area, dining, spa services, transportation, and any other \
guest needs.

## Guidelines
- Only answer questions related to the hotel and local area.
- For out-of-scope questions (politics, personal opinions, etc.), politely \
  deflect and redirect to hotel-related assistance.
- Use ONLY the information in the knowledge base. Do not make up facts.
- When a guest requests a booking or service, gather all required information \
  before calling the tool.
- Be concise and conversational — this is a voice interaction.
- Always confirm actions you've taken (e.g., "I've booked your reservation").
"""


def _build_hotel_turns(rng: random.Random, num_turns: int) -> List[VoiceTurn]:
    """Build a deterministic hotel concierge conversation."""
    from datetime import date, timedelta

    today = date.today()
    tomorrow = today + timedelta(days=1)
    day_after = today + timedelta(days=2)
    # Find next Wednesday
    days_until_wed = (2 - today.weekday()) % 7
    if days_until_wed == 0:
        days_until_wed = 7
    next_wednesday = today + timedelta(days=days_until_wed)

    today_str = today.isoformat()
    tomorrow_str = tomorrow.isoformat()
    wednesday_str = next_wednesday.isoformat()

    # Fixed conversation structure that exercises all capabilities
    all_turns = [
        # Turn 0: Greeting
        VoiceTurn(
            input="Hi there! I just checked into room 412. Can you tell me about the hotel restaurants?",
            golden_text="Welcome to the Grand Horizon Hotel! We have three dining options: "
                        "The Horizon Grill on the first floor serves American contemporary cuisine "
                        "for breakfast, lunch, and dinner. Sakura Lounge on the rooftop has Japanese "
                        "small plates and cocktails, open from 4 PM to midnight except Mondays. "
                        "And Cafe Mornings in the lobby is great for coffee and pastries until 3 PM.",
        ),
        # Turn 1: Knowledge grounding — specific detail
        VoiceTurn(
            input="What's the signature dish at the Horizon Grill? And what time does dinner start?",
            golden_text="The signature dish is the pan-seared salmon with citrus glaze, priced at $38. "
                        "Dinner service starts at 5:30 PM and runs until 10:00 PM.",
        ),
        # Turn 2: Restaurant booking (tool call)
        VoiceTurn(
            input="That sounds great. Can you book a table for two at the Horizon Grill "
                  "tonight at 7 PM? The name is Sarah Chen.",
            golden_text="I'll book that for you right away — a table for two at the Horizon Grill "
                        "tonight at 7:00 PM under Sarah Chen.",
            required_function_call={
                "name": "book_restaurant",
                "args": {
                    "guest_name": "Sarah Chen",
                    "restaurant": "The Horizon Grill",
                    "date": today_str,
                    "time": "19:00",
                    "party_size": 2,
                },
            },
            function_call_response={
                "status": "confirmed",
                "confirmation_id": "RES-7821",
                "details": "Table for 2 at The Horizon Grill, 7:00 PM",
            },
        ),
        # Turn 3: Spa inquiry (knowledge grounding)
        VoiceTurn(
            input="Thanks! What spa services do you offer? I'd love a massage.",
            golden_text="Our Serenity Spa on Floor 3 offers several services: Swedish massage "
                        "at $120 for 60 minutes, deep tissue massage at $140 for 60 minutes, "
                        "facials at $95 for 45 minutes, and hot stone massage at $160 for 90 minutes. "
                        "The spa is open from 7 AM to 9 PM daily.",
        ),
        # Turn 4: Spa booking (tool call)
        VoiceTurn(
            input="I'll take the deep tissue massage tomorrow at 10 AM please. Same name, Sarah Chen.",
            golden_text="I've booked a deep tissue massage for you tomorrow at 10:00 AM.",
            required_function_call={
                "name": "book_spa",
                "args": {
                    "guest_name": "Sarah Chen",
                    "service": "Deep tissue massage",
                    "date": tomorrow_str,
                    "time": "10:00",
                },
            },
            function_call_response={
                "status": "confirmed",
                "confirmation_id": "SPA-3344",
            },
        ),
        # Turn 5: Local area question (knowledge grounding)
        VoiceTurn(
            input="What are some things to do near the hotel?",
            golden_text="There's plenty nearby! Fisherman's Wharf is about a 15-minute walk. "
                        "Chinatown is just a 5-minute walk away. The Golden Gate Bridge is "
                        "about 15 minutes by car. You can catch a cable car one block east on "
                        "Powell Street. And Union Square for shopping is about half a mile south.",
        ),
        # Turn 6: Specific attraction detail
        VoiceTurn(
            input="How do I get to Alcatraz Island?",
            golden_text="You can take a ferry from Pier 33. I'd recommend booking at least "
                        "two weeks in advance since it's a very popular attraction.",
        ),
        # Turn 7: Out-of-scope question (deflection test)
        VoiceTurn(
            input="By the way, what do you think about the current political situation?",
            golden_text="I appreciate your curiosity, but I'm best suited to help with "
                        "hotel and local area questions. Is there anything else I can help "
                        "you with regarding your stay?",
        ),
        # Turn 8: Transportation inquiry
        VoiceTurn(
            input="I need to get to the airport on Wednesday morning. What are my options?",
            golden_text="We have an airport shuttle to SFO that runs every 60 minutes at $25 per "
                        "person, departing from the lobby. We also offer a complimentary town car "
                        "service within a 3-mile radius, subject to availability. And there's a "
                        "designated Uber/Lyft pickup area at the Pacific Avenue side entrance.",
        ),
        # Turn 9: Transportation booking (tool call)
        VoiceTurn(
            input="Let's go with the airport shuttle on Wednesday at 8 AM. Name is Sarah Chen.",
            golden_text="I've scheduled the airport shuttle for you on Wednesday at 8:00 AM.",
            required_function_call={
                "name": "schedule_transportation",
                "args": {
                    "guest_name": "Sarah Chen",
                    "service_type": "shuttle",
                    "date": wednesday_str,
                    "time": "08:00",
                    "destination": "SFO",
                },
            },
            function_call_response={
                "status": "confirmed",
                "pickup_location": "Hotel lobby",
                "estimated_arrival": "8:45 AM at SFO",
            },
        ),
        # Turn 10: Room issue — maintenance request (tool call)
        VoiceTurn(
            input="Oh, one more thing — the air conditioning in my room seems to be making "
                  "a weird rattling noise. Can someone take a look?",
            golden_text="I'm sorry about that. I'll submit a maintenance request right away.",
            required_function_call={
                "name": "submit_maintenance_request",
                "args": {
                    "room_number": "412",
                    "issue_description": "Air conditioning making a rattling noise",
                    "urgency": "medium",
                },
            },
            function_call_response={
                "status": "submitted",
                "ticket_id": "MNT-1102",
                "estimated_response": "Within 2 hours",
            },
        ),
        # Turn 11: WiFi question (knowledge grounding)
        VoiceTurn(
            input="What's the WiFi password again?",
            golden_text="The WiFi network is 'GrandHorizon-Guest' and the password is 'welcome2025'.",
        ),
        # Turn 12: Room service order (tool call)
        VoiceTurn(
            input="I'd like to order room service. Can I get a club sandwich and a cappuccino "
                  "to room 412?",
            golden_text="I'll place that order for you — a club sandwich and a cappuccino "
                        "to room 412. Delivery usually takes 25 to 40 minutes.",
            required_function_call={
                "name": "request_room_service",
                "args": {
                    "room_number": "412",
                    "items": "club sandwich, cappuccino",
                },
            },
            function_call_response={
                "status": "confirmed",
                "order_id": "RS-5567",
                "estimated_delivery": "35 minutes",
            },
        ),
        # Turn 13: Happy hour knowledge check
        VoiceTurn(
            input="Is there a happy hour at the Sakura Lounge?",
            golden_text="Yes! Happy hour at the Sakura Lounge is from 4:00 PM to 6:00 PM with "
                        "20% off all drinks. Just a heads-up, it's first-come first-served — "
                        "no reservations.",
        ),
        # Turn 14: Policy question (knowledge grounding)
        VoiceTurn(
            input="I'm thinking of having a friend visit. Is there an extra guest fee?",
            golden_text="Yes, there's a $40 per night fee for more than two guests in a standard room.",
        ),
        # Turn 15: Pet policy
        VoiceTurn(
            input="Can my friend bring their dog?",
            golden_text="Dogs under 30 pounds are welcome with a $75 per night pet fee. "
                        "The hotel is otherwise a non-smoking property with quiet hours "
                        "from 10 PM to 7 AM.",
        ),
        # Turn 16: Late checkout
        VoiceTurn(
            input="Can I get a late checkout on Wednesday?",
            golden_text="Late check-out is available until 2:00 PM for a $50 fee. "
                        "Standard check-out is at 11:00 AM. Would you like me to arrange that?",
        ),
        # Turn 17: Parking question
        VoiceTurn(
            input="What are the parking options?",
            golden_text="We offer valet parking at $45 per night and self-parking in the "
                        "garage at $30 per night.",
        ),
        # Turn 18: Pool/fitness question
        VoiceTurn(
            input="Is there a pool? And a gym?",
            golden_text="Yes! There's a heated outdoor pool on Floor 4, open from 6 AM to 10 PM. "
                        "The fitness center is on Floor 3 and is accessible 24/7 with your room key.",
        ),
        # Turn 19: Ending conversation (tool call)
        VoiceTurn(
            input="That's everything, thanks so much! You've been super helpful.",
            golden_text="You're welcome, Sarah! Enjoy your stay at the Grand Horizon. "
                        "Don't hesitate to call if you need anything else.",
            required_function_call={
                "name": "end_session",
                "args": {},
            },
            function_call_response={"status": "session_ended"},
        ),
    ]

    # Return requested number of turns (cap at available)
    count = min(num_turns, len(all_turns))
    return all_turns[:count]


def generate_builtin_hotel(
    num_turns: int = 20,
    seed: int = 42,
    scenario_id: Optional[str] = None,
) -> VoiceScenario:
    """
    Generate a deterministic hotel concierge scenario.

    Args:
        num_turns: Number of turns (max 20).
        seed: Random seed for reproducibility.
        scenario_id: Custom scenario ID.

    Returns:
        A complete VoiceScenario.
    """
    rng = random.Random(seed)
    turns = _build_hotel_turns(rng, num_turns)

    return VoiceScenario(
        id=scenario_id or "voice_hotel_concierge_001",
        domain="hotel_concierge",
        system_prompt=HOTEL_SYSTEM_PROMPT,
        knowledge_base=HOTEL_KNOWLEDGE_BASE,
        tools=HOTEL_TOOLS,
        turns=turns,
    )


def list_builtin_domains() -> List[Dict[str, str]]:
    """List available built-in voice domains."""
    return [
        {
            "name": "hotel_concierge",
            "description": "Hotel front desk assistant — restaurant bookings, "
                           "spa, transportation, room service, and local recommendations. "
                           "20 turns, 7 tool calls.",
            "turns": "20",
            "tool_calls": "7",
        },
    ]
