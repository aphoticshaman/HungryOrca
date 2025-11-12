"""
Quantum Tarot - API Test Script
Tests complete user flow from registration through reading
"""

import requests
import json
from datetime import datetime
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

# Color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title:^70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(label: str, value: Any):
    """Print info line"""
    print(f"{Colors.OKCYAN}{label}:{Colors.ENDC} {value}")


def test_health_check():
    """Test health check endpoint"""
    print_section("1. Health Check")

    response = requests.get(f"{BASE_URL}/health")

    if response.status_code == 200:
        data = response.json()
        print_success("API is operational!")
        print_info("Status", data['status'])
        print_info("Database", data['database'])
        print_info("Quantum Engine", data['quantum_engine'])
        return True
    else:
        print_error(f"Health check failed: {response.status_code}")
        return False


def create_test_user() -> Dict[str, Any]:
    """Create a test user"""
    print_section("2. Create Test User")

    user_data = {
        "name": "Luna Starlight",
        "email": f"luna.test.{int(datetime.now().timestamp())}@example.com",
        "birthday": "1995-06-15T00:00:00",
        "gender_identity": "female",
        "pronouns": "she/her"
    }

    response = requests.post(f"{BASE_URL}/users", json=user_data)

    if response.status_code == 201:
        user = response.json()
        print_success("User created successfully!")
        print_info("User ID", user['id'])
        print_info("Name", user['name'])
        print_info("Sun Sign", user['sun_sign'])
        print_info("Subscription", user['subscription_tier'])
        return user
    else:
        print_error(f"User creation failed: {response.status_code}")
        print_error(response.text)
        return None


def get_personality_questions(reading_type: str = "romance"):
    """Get personality questions"""
    print_section("3. Get Personality Questions")

    response = requests.get(f"{BASE_URL}/personality/questions/{reading_type}")

    if response.status_code == 200:
        data = response.json()
        print_success(f"Retrieved {len(data['questions'])} questions for {reading_type}")

        # Show first 3 questions
        for i, q in enumerate(data['questions'][:3], 1):
            print(f"\n{Colors.BOLD}Question {i}:{Colors.ENDC} {q['text']}")
            print(f"{Colors.OKCYAN}Options:{Colors.ENDC}")
            for opt in q['options']:
                print(f"  • {opt}")

        return data
    else:
        print_error(f"Failed to get questions: {response.status_code}")
        return None


def create_personality_profile(user_id: str, reading_type: str = "romance") -> Dict[str, Any]:
    """Create personality profile"""
    print_section("4. Create Personality Profile")

    # Sample responses (would come from user in real app)
    responses = {
        "romance_1": "Emotional intimacy and deep connection",
        "romance_2": "Address it immediately and directly",
        "romance_3": "4 - Quite a bit",
        "romance_4": "Spends quality time with you",
        "romance_5": "Communication breakdowns",
        "romance_6": "False",
        "romance_7": "Take things slowly and see what develops",
        "romance_8": "4 - Somewhat optimistic",
        "romance_9": "Emotionally expressive and open",
        "romance_10": "Focus on yourself and let love find you"
    }

    profile_data = {
        "user_id": user_id,
        "reading_type": reading_type,
        "responses": responses
    }

    response = requests.post(f"{BASE_URL}/personality/profiles", json=profile_data)

    if response.status_code == 201:
        profile = response.json()
        print_success("Personality profile created!")
        print_info("Profile ID", profile['id'])
        print_info("Communication Voice", profile['communication_voice'])
        print_info("Aesthetic Profile", profile['aesthetic_profile'])
        print_info("Primary Framework", profile['primary_framework'])
        return profile
    else:
        print_error(f"Profile creation failed: {response.status_code}")
        print_error(response.text)
        return None


def check_reading_limit(user_id: str):
    """Check if user can perform reading"""
    print_section("5. Check Reading Limit")

    response = requests.get(f"{BASE_URL}/users/{user_id}/reading-limit")

    if response.status_code == 200:
        data = response.json()
        print_info("Can Read", "✓ Yes" if data['can_read'] else "✗ No")
        print_info("Readings Today", data['readings_today'])
        print_info("Daily Limit", data['limit'])
        print_info("Subscription", data['subscription_tier'])
        return data['can_read']
    else:
        print_error(f"Failed to check limit: {response.status_code}")
        return False


def create_reading(user_id: str, reading_type: str = "romance", spread_type: str = "three_card") -> Dict[str, Any]:
    """Create a tarot reading - THE MAIN EVENT!"""
    print_section("6. Create Quantum Tarot Reading ✨")

    reading_data = {
        "user_id": user_id,
        "reading_type": reading_type,
        "spread_type": spread_type,
        "user_intention": "What do I need to know about my love life right now?"
    }

    print(f"{Colors.BOLD}Performing quantum reading...{Colors.ENDC}")
    print(f"🎯 Intention: {reading_data['user_intention']}")
    print(f"📖 Spread: {spread_type}")
    print(f"💫 Collapsing quantum superposition...\n")

    response = requests.post(f"{BASE_URL}/readings", json=reading_data)

    if response.status_code == 201:
        reading = response.json()
        print_success("Reading created successfully!")
        print_info("Reading ID", reading['id'])
        print_info("Type", reading['reading_type'])
        print_info("Spread", reading['spread_type'])
        print_info("Cards Drawn", len(reading['cards']))

        return reading
    else:
        print_error(f"Reading creation failed: {response.status_code}")
        print_error(response.text)
        return None


def display_reading(reading: Dict[str, Any]):
    """Display the complete reading beautifully"""
    print_section("7. Your Quantum Tarot Reading")

    if reading['user_intention']:
        print(f"{Colors.BOLD}Your Question:{Colors.ENDC} {reading['user_intention']}\n")

    for i, card in enumerate(reading['cards'], 1):
        print(f"{Colors.BOLD}{Colors.HEADER}{'─' * 70}{Colors.ENDC}")
        print(f"{Colors.BOLD}Position {i}: {card['position_name']}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'─' * 70}{Colors.ENDC}\n")

        # Card name with reversed indicator
        reversed = " (Reversed)" if card['is_reversed'] else ""
        print(f"{Colors.OKGREEN}{Colors.BOLD}🃏 {card['card_name']}{reversed}{Colors.ENDC}")

        # Keywords
        keywords_str = " • ".join(card['keywords'])
        print(f"{Colors.OKCYAN}Keywords: {keywords_str}{Colors.ENDC}\n")

        # Interpretation
        print(f"{Colors.BOLD}Message:{Colors.ENDC}")
        # Word wrap the interpretation
        words = card['interpretation_text'].split()
        line = ""
        for word in words:
            if len(line + word) > 65:
                print(f"  {line}")
                line = word + " "
            else:
                line += word + " "
        if line:
            print(f"  {line}")

        # Quantum signature
        print(f"\n{Colors.WARNING}Quantum Signature: {card['quantum_signature'][:16]}...{Colors.ENDC}")
        print()


def get_available_spreads():
    """Get available spread types"""
    print_section("Available Spreads")

    response = requests.get(f"{BASE_URL}/spreads")

    if response.status_code == 200:
        data = response.json()
        print_success(f"Found {len(data['spreads'])} spread types:")

        for spread in data['spreads']:
            premium = "⭐ PREMIUM" if spread['premium'] else "FREE"
            print(f"\n{Colors.BOLD}{spread['name']}{Colors.ENDC} ({premium})")
            print(f"  {spread['description']}")
            print(f"  {Colors.OKCYAN}Positions: {spread['positions']}{Colors.ENDC}")

        return data
    else:
        print_error(f"Failed to get spreads: {response.status_code}")
        return None


def main():
    """Run complete test flow"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║               QUANTUM TAROT API - TEST SUITE                      ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(Colors.ENDC)

    try:
        # 1. Health check
        if not test_health_check():
            print_error("\n❌ API is not running. Start the server first:")
            print_error("   cd backend/api && python main.py")
            return

        # 2. Create user
        user = create_test_user()
        if not user:
            print_error("\n❌ Failed to create user. Check API logs.")
            return

        user_id = user['id']

        # 3. Get personality questions
        questions = get_personality_questions("romance")
        if not questions:
            print_error("\n❌ Failed to get questions.")
            return

        # 4. Create personality profile
        profile = create_personality_profile(user_id, "romance")
        if not profile:
            print_error("\n❌ Failed to create profile.")
            return

        # 5. Check reading limit
        can_read = check_reading_limit(user_id)
        if not can_read:
            print_error("\n❌ User cannot read (daily limit reached).")
            return

        # 6. Create reading
        reading = create_reading(user_id, "romance", "three_card")
        if not reading:
            print_error("\n❌ Failed to create reading.")
            return

        # 7. Display reading
        display_reading(reading)

        # 8. Show available spreads
        get_available_spreads()

        # Success!
        print_section("🎉 Test Complete!")
        print_success("All tests passed successfully!")
        print()
        print(f"{Colors.BOLD}Your quantum tarot API is fully operational.{Colors.ENDC}")
        print(f"{Colors.OKCYAN}API Documentation: {BASE_URL}/docs{Colors.ENDC}")
        print()
        print(f"{Colors.WARNING}Next steps:{Colors.ENDC}")
        print("  1. Build the React Native mobile app")
        print("  2. Generate card artwork (78 cards)")
        print("  3. Integrate payment processing")
        print("  4. Deploy to production")
        print()

    except requests.exceptions.ConnectionError:
        print_error("\n❌ Cannot connect to API. Make sure the server is running:")
        print_error("   cd backend/api && python main.py")
    except Exception as e:
        print_error(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
