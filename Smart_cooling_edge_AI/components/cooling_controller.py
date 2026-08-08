from pathlib import Path
from Smart_cooling_edge_AI.utils.configurations import read_yaml
from Smart_cooling_edge_AI.logging.logger import get_logger

# logger initialize
logger = get_logger("CoolingController")

class CoolingController:
    def __init__(self):
        try:
            project_root = Path(__file__).resolve().parent.parent.parent
            config = read_yaml(project_root / "params.yaml")
            logger.info("Configuration parameters loaded successfully in CoolingController.")
            
            # Extract configuration parameters safely with fallback defaults to ensure production resilience and prevent KeyError crashes
            self.rules = config.get("AC_CONTROL_RULES", {})
            if not self.rules:
                logger.warning("AC_CONTROL_RULES is empty or missing in params.yaml.")
                
        except Exception as e:
            logger.critical(f"Failed to initialize CoolingController configuration: {str(e)}", exc_info=True)
            self.rules = {}



    def regulate_cooling_system(self, student_count: int) -> dict:
        """
        Controls 4 AC units, temperatures, fan speeds, and louvers dynamically 
        based on the rules defined in params.yaml.
        Returns structured data dictionaries.
        """
        try:
            logger.info(f"Regulating cooling system for detected student count: {student_count}")
            
            if not self.rules:
                logger.error("No cooling rules available. Returning default fallback HVAC state.")
                return self._get_default_fallback_state()
            
            # Loop through the rules defined in YAML to match student count range
            for rule_key, rule in self.rules.items():
                min_s = rule.get("min_students", 0)
                max_s = rule.get("max_students", float('inf'))
                
                if min_s <= student_count <= max_s:
                    logger.info(f"Matched rule '{rule_key}' for student count {student_count} (Range: {min_s}-{max_s}).")

                    return {
                        "occupancy_level": rule["occupancy_level"],
                        "ac_status_text": rule["ac_status_text"],
                        "ac_active_count": rule["ac_active_count"],
                        "ac_mode": rule["ac_mode"],
                        "temp_min": float(rule["temp_min"]),
                        "temp_max": float(rule["temp_max"]),
                        "fan_speed": rule["fan_speed"],
                        "louver_position": rule["louver_position"],
                        "description": rule["description"]
                    }
            
            # Fallback if student count exceeds the highest defined range
            highest_rule_key = list(self.rules.keys())[-1]
            highest_rule = self.rules[highest_rule_key]
            logger.warning(f"Student count {student_count} exceeds defined ranges. Applying highest rule '{highest_rule_key}'.")

            return {
                "occupancy_level": highest_rule["occupancy_level"],
                "ac_status_text": highest_rule["ac_status_text"],
                "ac_active_count": highest_rule["ac_active_count"],
                "ac_mode": highest_rule["ac_mode"],
                "temp_min": float(highest_rule["temp_min"]),
                "temp_max": float(highest_rule["temp_max"]),
                "fan_speed": highest_rule["fan_speed"],
                "louver_position": highest_rule["louver_position"],
                "description": highest_rule["description"]
            }
        except Exception as e:
            logger.error(f"Error occurred during cooling system regulation: {str(e)}", exc_info=True)
            return self._get_default_fallback_state()



    # Provide a safe default fallback configuration to ensure system continuity and prevent crashes during exceptions
    def _get_default_fallback_state(self) -> dict:
            """Returns a safe hardcoded fallback state in case of extreme errors."""

            return {
                "occupancy_level": "ERROR_FALLBACK",
                "ac_status_text": "SYSTEM_ERROR",
                "ac_active_count": 0,
                "ac_mode": "OFF",
                "temp_min": 24.0,
                "temp_max": 24.0,
                "fan_speed": "LOW",
                "louver_position": "CENTER",
                "description": "Fallback state triggered due to exception."
            }


'''

====old======

    def regulate_cooling_system(self, student_count: int) -> dict:
        """
        Controls 4 AC units, temperatures, fan speeds, and louvers based on the Smart Classroom Operation Matrix.
        Returns structured data dictionaries.
        """
        if 0 <= student_count <= 20:
            hvac_data = {
                "occupancy_level": "low",
                "ac_status_text": "AC 2 ON (Alternate)",
                "ac_active_count": 2,
                "ac_mode": "alternate",
                "temp_min": 24.0,
                "temp_max": 25.0,
                "fan_speed": "low",
                "louver_position": "down",
                "description": "LOW (Pre-lecture / Small group)"
            }
        elif 21 <= student_count <= 60:
            hvac_data = {
                "occupancy_level": "medium",
                "ac_status_text": "AC 4 ON (Normal)",
                "ac_active_count": 4,
                "ac_mode": "normal",
                "temp_min": 22.0,
                "temp_max": 23.0,
                "fan_speed": "medium",
                "louver_position": "swing",
                "description": "MEDIUM (Half filled)"
            }
        elif 61 <= student_count <= 110:
            hvac_data = {
                "occupancy_level": "high",
                "ac_status_text": "AC 4 ON (Active)",
                "ac_active_count": 4,
                "ac_mode": "active",
                "temp_min": 20.0,
                "temp_max": 21.0,
                "fan_speed": "high",
                "louver_position": "horizontal",
                "description": "HIGH (Well filled)"
            }
        else: # 111 - 150+
            hvac_data = {
                "occupancy_level": "full",
                "ac_status_text": "AC 4 ON (Full Power / Turbo)",
                "ac_active_count": 4,
                "ac_mode": "turbo",
                "temp_min": 17.0,
                "temp_max": 19.0,
                "fan_speed": "max",
                "louver_position": "all_swing_fresh_air",
                "description": "FULL (Maximum capacity)"
            }

    return hvac_data

'''
                
        