import os
from openai import AzureOpenAI
import logging
import json

logger = logging.getLogger(__name__)

def generate_explanation(predictions):
    """Generate a natural language explanation of the referee decision"""
    # Format predictions into a readable format for the prompt
    prediction_text = ""
    
    # Access as Pydantic model (not as dictionary)
    for pred in predictions:
        category = pred.category
        # Each category has a list of details, get the first one
        detail = pred.details[0]
        prediction = detail.prediction
        probability = detail.probability
        
        prediction_text += f"{category}: {prediction} (confidence: {probability:.1%})\n"
    
    try:
        # Try to use Azure OpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
        
        # Enhanced prompts for better response format
        system_prompt = """You are a football referee expert. Provide a concise, natural-sounding explanation.
        Your response must be a valid JSON object with these fields:
        - decision: A concise decision (e.g., "Yellow Card", "No Card", "Red Card")
        - explanation: A natural, conversational explanation that sounds like a referee or commentator
        
        Keep the explanation concise but insightful, avoiding formulaic language."""
        
        user_prompt = f"""Based on these predictions, explain the referee decision:

{prediction_text}

Respond with a JSON object containing 'decision' and 'explanation' fields.
Make your explanation sound natural and authoritative, like an experienced referee or commentator would explain it.
"""
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        logger.error(f"Error with Azure OpenAI: {str(e)}")
        
        # Fallback to simple explanation in the new format
        severity = "No Card"
        action = "Normal Play"
        try_to_play = "Yes"
        
        # Extract main fields we need for fallback
        for pred in predictions:
            if pred.category == "Severity":
                severity_text = pred.details[0].prediction.lower()
                if "yellow" in severity_text:
                    severity = "Yellow Card"
                elif "red" in severity_text:
                    severity = "Red Card"
            elif pred.category == "Action Class":
                action = pred.details[0].prediction
            elif pred.category == "Try to Play":
                try_to_play = pred.details[0].prediction
        
        # Simple fallback explanation in structured format
        if severity == "No Card":
            explanation = f"The action was deemed legal with no serious infringement."
        else:
            explanation = f"{action} infringement {'with' if try_to_play == 'Yes' else 'without'} attempt to play the ball."
            
        return {
            "decision": severity,
            "explanation": explanation
        }