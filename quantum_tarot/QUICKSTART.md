# Quantum Tarot - Quick Start Guide

## Prerequisites
- Python 3.9 or higher
- pip (Python package installer)
- (Optional) Virtual environment tool (venv, conda, etc.)

## Installation

### 1. Create Virtual Environment (Recommended)
```bash
cd quantum_tarot
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Initialize Database
```bash
cd backend
python -c "from database.schema import init_db; init_db()"
```

This creates a SQLite database file `quantum_tarot.db` in the backend directory.

## Running the API Server

### Start the server:
```bash
cd backend/api
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

### Interactive API Documentation
Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing the API

### 1. Health Check
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "quantum_engine": "operational",
  "timestamp": "2024-..."
}
```

### 2. Create a User
```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Luna Starlight",
    "email": "luna@example.com",
    "birthday": "1995-06-15T00:00:00",
    "pronouns": "she/her"
  }'
```

Save the returned `id` - you'll need it for subsequent requests.

### 3. Get Attunement Questions
```bash
curl http://localhost:8000/personality/questions/romance
```

### 4. Create Personality Profile
```bash
curl -X POST http://localhost:8000/personality/profiles \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "YOUR_USER_ID_HERE",
    "reading_type": "romance",
    "responses": {
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
  }'
```

### 5. Create a Reading (THE MAIN EVENT!)
```bash
curl -X POST http://localhost:8000/readings \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "YOUR_USER_ID_HERE",
    "reading_type": "romance",
    "spread_type": "three_card",
    "user_intention": "What do I need to know about my love life right now?"
  }'
```

This will return a complete reading with:
- Quantum-selected cards
- Personalized interpretations adapted to the user's communication style
- Position meanings
- Keywords
- Quantum signatures for provenance

## API Endpoints Reference

### Users
- `POST /users` - Create new user
- `GET /users/{user_id}` - Get user details
- `GET /users/{user_id}/reading-limit` - Check if user can read today

### Personality Profiles
- `GET /personality/questions/{reading_type}` - Get attunement questions
- `POST /personality/profiles` - Create personality profile
- `GET /users/{user_id}/personality-profiles` - Get all profiles for user

### Readings
- `POST /readings` - Create new reading (main endpoint!)
- `GET /readings/{reading_id}` - Get specific reading
- `GET /users/{user_id}/readings` - Get reading history (premium)
- `PATCH /readings/{reading_id}/favorite` - Toggle favorite (premium)

### Utility
- `GET /spreads` - List available spread types
- `GET /reading-types` - List available reading types
- `GET /health` - Health check

## Using the Automated Test Script

We've provided a test script that walks through the entire flow:

```bash
cd backend
python test_api.py
```

This will:
1. Create a test user
2. Get personality questions
3. Submit responses and create profile
4. Perform a complete tarot reading
5. Retrieve the reading
6. Display everything beautifully

## Environment Variables

Create a `.env` file in the `quantum_tarot` directory:

```env
# Database
DATABASE_URL=sqlite:///quantum_tarot.db
# For PostgreSQL: postgresql://user:password@localhost/quantum_tarot

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Quantum Engine
USE_EXTERNAL_QUANTUM=True  # Set to False to disable external quantum API

# Security (for production)
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Production Deployment

### Using Docker (Recommended)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t quantum-tarot .
docker run -p 8000:8000 quantum-tarot
```

### Using Heroku
```bash
# Install Heroku CLI
heroku create quantum-tarot-api

# Set environment variables
heroku config:set DATABASE_URL=your-postgresql-url

# Deploy
git push heroku main
```

### Using Railway/Render
Both platforms auto-detect FastAPI apps. Just connect your GitHub repo and deploy!

## Database Migrations (Alembic)

Initialize migrations:
```bash
cd backend
alembic init migrations
```

Create migration:
```bash
alembic revision --autogenerate -m "Initial schema"
```

Apply migrations:
```bash
alembic upgrade head
```

## Troubleshooting

### ImportError: No module named 'fastapi'
Make sure you've activated your virtual environment and installed dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Database errors
Delete the existing database and recreate:
```bash
rm backend/quantum_tarot.db
python -c "from database.schema import init_db; init_db()"
```

### Port already in use
Change the port in the run command:
```bash
uvicorn main:app --port 8001
```

### Quantum API timeout
The quantum random number API might be slow/unavailable. The system falls back to OS-level crypto random, which is still high-quality. You can disable external quantum sources:
```bash
export USE_EXTERNAL_QUANTUM=False
```

## Next Steps

1. **Test thoroughly** - Run through all endpoints
2. **Add authentication** - Implement JWT tokens for production
3. **Set up monitoring** - Add logging and error tracking
4. **Build mobile app** - Use React Native to consume this API
5. **Deploy to production** - Use a proper database (PostgreSQL) and hosting

## Support

For issues or questions:
- Check the API docs: http://localhost:8000/docs
- Review the code comments in `backend/api/main.py`
- Test with the provided `test_api.py` script

---

**You now have a fully functional quantum tarot backend API!** 🎯✨

The hard part (quantum randomness, psychology integration, adaptive language) is done. Now it's just connecting a mobile UI to these endpoints.
