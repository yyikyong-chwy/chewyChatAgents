from __future__ import annotations

from .petState import PetProfile
from .customerState import Customer
from .bundleState import bundleState
from .ChewyJourneyChatState import ChewyJourneyChatState
from .sessionState import SessionStateModel
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # ...\LG_Agents
sys.path.insert(0, str(ROOT))
import helperFunctions.geo_utils as geo_utils
#from helperFunctions.geo_utils import zip_to_latlon, WeatherTool

# ---------- Example usage ----------
def build_customer_profile(customer_id: int) -> Customer:
    if customer_id == 1:
        return Customer(
            id="1",
            first_name="John",
            last_name="Doe",
            email="john.doe@example.com",
            phone="1234567890",
            preferred_units="imperial",
            timezone="America/Los_Angeles",
            zip_code="98075", #Seattle, WA,
        )
    elif customer_id == 2:
        return Customer(
            id="2",
            first_name="Janet",
            last_name="Enver",
            email="janet.enver@example.com",
            phone="0987654321",
            preferred_units="metric",
            zip_code="78745", #Austin, TX
        )
    elif customer_id == 3:
        return Customer(
            id="3",
            first_name="Abraham",
            last_name="Lincoln",
            email="abraham.lincoln@example.com",
            phone="1234567890",
            zip_code="32013", #Jacksonville, FL
        )                    
    elif customer_id == 4:
        return Customer(
            id="4",
            first_name="Elvis",
            last_name="Presley",
            email="elvis.presley@example.com",
            phone="1234567890",
            zip_code="30316", #Atlanta, GA
        )
    elif customer_id == 5:
        return Customer(
            id="5",
            first_name="Marie",
            last_name="Curie",
            email="marie.curie@example.com",
            phone="1234567890",
            zip_code="90001", #Los Angeles, CA
        )
    elif customer_id == 6:
        return Customer(
            id="6",
            first_name="Sophia",
            last_name="Maria",
            email="sophia.maria@example.com",
            phone="1234567890",
            zip_code="97027", #Portland, OR
        )
    else:
        raise ValueError("Invalid customer ID. Choose 1, 2, 3, 4, 5, or 6.")




def build_pet_profile(pet_id: int) -> PetProfile:
    if pet_id == 1:
        return PetProfile(
            species="dog",
            breed="Golden Retriever",
            gender="male",
            pet_name = "Max",
            age_months=4,
            weight_lb=18.5,
            habits=["chewer", "crate-trained", "likes to roll in mud", "destroyed couch", "loves to chew on shoes"],
            recent_conditions=["healthy", "ticks", "ordor", "outdoor", "does not play well with other dogs", "underweight"],
            geo_eventcondition=["heavy rain predicted in coming 2 weeks", "Halloween in coming 2 weeks"],
            recent_purchases=["kibble", "leash", "bed", "toy"]

        )
    elif pet_id == 2:
        return PetProfile(
            species="dog",
            breed="French Bulldog",
            gender="female",
            pet_name = "Bella",
            age_months=12,
            weight_lb=27.0,
            habits=["sensitive_skin", "indoor"],
            recent_conditions=["overweight", "sleeping disorder", "timid and shy"],
            geo_eventcondition=["heatwave predicted in coming 2 weeks", "bring pet to beer festival in 10 days"],
            recent_purchases=["kibble", "leash", "bed", "toy"]

        )
    elif pet_id == 3:
        return PetProfile(
            species="cat",
            breed="Maine Coon",
            gender="female",
            pet_name = "Luna",
            age_months=1,
            weight_lb=1.5,
            habits=["long-hair", "indoor", "likes to play with string"],
            recent_conditions=["healthy", "clean", "fights with other cats", "easily startled"],
            geo_eventcondition=["mild weather"],
            recent_purchases=["kibble", "leash", "bed", "toy"]

        )
    elif pet_id == 4:
        return PetProfile(
            species="dog",
            breed="Doberman",
            gender="male",
            pet_name = "Bruni",
            age_months=5,
            weight_lb=49.2,
            habits=["aggressive", "teething", "hyperactive"],            
            recent_conditions=["healthy", "clean", "chew on shoes", "barks alot"],
            geo_eventcondition=["winter storm predicted in coming 2 weeks", "Christmas in 3 weeks"]
        )
    elif pet_id == 5:
        return PetProfile(
            species="cat",
            breed="Siamese",
            gender="male",
            pet_name = "Leo",
            age_months=4,
            weight_lb=4.3,
            habits=["indoor", "talkative", "clingy", "scratches furniture"],
            recent_conditions=["hairball issues", "healthy", "sensitive stomach"],
            geo_eventcondition=["foggy mornings common in next 2 weeks", "Thanksgiving holiday in 1 week"],
            recent_purchases=["premium kibble", "cat tree", "scratching post", "treats"]
        )
    elif pet_id == 6:
        return PetProfile(
            species="dog",
            breed="German Shepherd",
            gender="female",
            pet_name = "Roxy",
            age_months=2.5,
            weight_lb=18,
            habits=["guard-dog", "digging holes", "active", "fetch-loving", "does not eat chicken"],
            recent_conditions=["joint pain", "shedding heavily", "heat sensitivity", "overweight"],
            geo_eventcondition=["extreme heat warning in next 2 weeks", "4th of July fireworks in 10 days"],
            recent_purchases=["joint supplements", "cooling mat", "dog food", "training collar"]
        )
    else:
        raise ValueError("Invalid pet ID. Choose 1, 2, 3, 4, 5, or 6.")  

# ----------------------------
# Loader function
# ----------------------------

def load_langgraph_state_session(customer_id: int) -> SessionStateModel:
    customer = build_customer_profile(customer_id)
    customer.geo = geo_utils.zip_to_latlon(customer.zip_code)
    customer.weather_now = geo_utils.WeatherTool.get_current_from_zip(customer.zip_code)
    pet = build_pet_profile(customer_id)
    return SessionStateModel(
        customer=customer,
        pet_profile=pet,
        bundle_state=bundleState(), # I shall leave this blank for now, but will be populated by the bundle_creator_agent
        chewy_journey_chat_state=ChewyJourneyChatState(),
    )

#Testing
if __name__ == "__main__":
    print(load_langgraph_state_session(1))
    print(load_langgraph_state_session(2))
    print(load_langgraph_state_session(3))
    print(load_langgraph_state_session(4))
    print(load_langgraph_state_session(5))
    print(load_langgraph_state_session(6))
