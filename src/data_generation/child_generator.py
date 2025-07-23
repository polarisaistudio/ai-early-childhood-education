"""
Child Profile Generator for AI Early Childhood Education System

This module generates realistic synthetic child profiles for testing and development.
"""

import uuid
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from pathlib import Path
import json

# Import configuration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import AGE_RANGES, ATTENTION_SPANS, LEARNING_STYLES, DEVELOPMENTAL_PACES, INTERESTS


class ChildProfileGenerator:
    """Generates realistic synthetic child profiles with developmental characteristics."""
    
    def __init__(self, seed: int = None):
        """
        Initialize the generator.
        
        Args:
            seed: Random seed for reproducible results
        """
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        self.age_ranges = AGE_RANGES
        self.attention_spans = ATTENTION_SPANS
        self.learning_styles = LEARNING_STYLES
        self.developmental_paces = DEVELOPMENTAL_PACES
        self.interests = INTERESTS
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        return obj
    
    def generate_child(self) -> Dict:
        """
        Generate a single child profile.
        
        Returns:
            Dictionary containing child profile data
        """
        # Generate basic demographics
        age_months = int(np.random.randint(12, 97))  # 1-8 years
        stage = self._determine_stage(age_months)
        
        # Generate developmental characteristics
        developmental_pace = np.random.choice(
            self.developmental_paces, 
            p=[0.15, 0.7, 0.15]  # Most children develop typically
        )
        
        # Generate interests (2-4 interests per child)
        num_interests = np.random.randint(2, 5)
        child_interests = np.random.choice(
            self.interests, 
            size=num_interests, 
            replace=False
        ).tolist()
        
        # Generate learning preferences
        learning_style = np.random.choice(self.learning_styles)
        attention_span = self._generate_attention_span(stage, developmental_pace)
        
        # Generate skill levels based on age and developmental pace
        skill_levels = self._generate_skill_levels(age_months, developmental_pace)
        
        # Create profile
        profile = {
            'id': str(uuid.uuid4()),
            'created_at': datetime.now().isoformat(),
            'age_months': age_months,
            'age_years': float(round(age_months / 12, 1)),
            'developmental_stage': stage,
            'developmental_pace': developmental_pace,
            'interests': child_interests,
            'learning_style': learning_style,
            'attention_span_minutes': attention_span,
            'skill_levels': skill_levels,
            'special_considerations': self._generate_special_considerations(),
            'family_context': self._generate_family_context()
        }
        
        return self._convert_numpy_types(profile)
    
    def generate_cohort(self, n_children: int = 100) -> List[Dict]:
        """
        Generate a cohort of children.
        
        Args:
            n_children: Number of children to generate
            
        Returns:
            List of child profiles
        """
        cohort = []
        
        print(f"Generating {n_children} child profiles...")
        for i in range(n_children):
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{n_children} profiles")
            
            child = self.generate_child()
            cohort.append(child)
        
        print("✅ Cohort generation complete!")
        return cohort
    
    def _determine_stage(self, age_months: int) -> str:
        """Determine developmental stage based on age."""
        if age_months < 36:
            return "toddler"
        elif age_months < 60:
            return "preschool"
        else:
            return "early_elementary"
    
    def _generate_attention_span(self, stage: str, developmental_pace: str) -> int:
        """Generate realistic attention span based on stage and pace."""
        base_min, base_max = self.attention_spans[stage]
        
        # Adjust based on developmental pace
        if developmental_pace == "fast":
            modifier = 1.2
        elif developmental_pace == "slow":
            modifier = 0.8
        else:
            modifier = 1.0
        
        adjusted_min = int(base_min * modifier)
        adjusted_max = int(base_max * modifier)
        
        return int(np.random.randint(adjusted_min, adjusted_max + 1))
    
    def _generate_skill_levels(self, age_months: int, developmental_pace: str) -> Dict:
        """Generate skill levels across different domains."""
        
        # Base skill progression by age (0-100 scale)
        base_skills = {
            'gross_motor': min(100, max(0, (age_months - 12) * 1.2 + np.random.normal(0, 10))),
            'fine_motor': min(100, max(0, (age_months - 12) * 1.1 + np.random.normal(0, 10))),
            'language': min(100, max(0, (age_months - 12) * 1.3 + np.random.normal(0, 12))),
            'cognitive': min(100, max(0, (age_months - 12) * 1.0 + np.random.normal(0, 10))),
            'social_emotional': min(100, max(0, (age_months - 12) * 1.1 + np.random.normal(0, 8))),
            'pre_literacy': min(100, max(0, (age_months - 24) * 1.5 + np.random.normal(0, 15))),
            'pre_math': min(100, max(0, (age_months - 24) * 1.4 + np.random.normal(0, 12)))
        }
        
        # Adjust based on developmental pace
        pace_multipliers = {
            'slow': 0.8,
            'typical': 1.0,
            'fast': 1.2
        }
        
        multiplier = pace_multipliers[developmental_pace]
        
        adjusted_skills = {}
        for skill, value in base_skills.items():
            adjusted_value = value * multiplier
            # Add some individual variation
            adjusted_value += np.random.normal(0, 5)
            # Keep within bounds and convert to native Python float
            adjusted_skills[skill] = float(max(0, min(100, round(adjusted_value, 1))))
        
        return adjusted_skills
    
    def _generate_special_considerations(self) -> List[str]:
        """Generate any special considerations (10% of children)."""
        considerations = []
        
        if np.random.random() < 0.1:  # 10% have special considerations
            possible_considerations = [
                'speech_delay', 'motor_delay', 'attention_concerns',
                'sensory_processing', 'social_anxiety', 'giftedness'
            ]
            
            num_considerations = np.random.choice([1, 2], p=[0.8, 0.2])
            considerations = np.random.choice(
                possible_considerations, 
                size=num_considerations, 
                replace=False
            ).tolist()
        
        return considerations
    
    def _generate_family_context(self) -> Dict:
        """Generate family context information."""
        return {
            'household_size': int(np.random.choice([2, 3, 4, 5, 6], p=[0.1, 0.3, 0.4, 0.15, 0.05])),
            'siblings': int(np.random.choice([0, 1, 2, 3], p=[0.2, 0.5, 0.25, 0.05])),
            'primary_language': str(np.random.choice(['English', 'Spanish', 'Other'], p=[0.7, 0.2, 0.1])),
            'caregiver_education': str(np.random.choice(
                ['high_school', 'some_college', 'bachelors', 'graduate'], 
                p=[0.2, 0.3, 0.35, 0.15]
            )),
            'childcare_setting': str(np.random.choice(
                ['home', 'family_daycare', 'center_based', 'preschool'], 
                p=[0.3, 0.2, 0.3, 0.2]
            ))
        }
    
    def save_cohort(self, cohort: List[Dict], filename: str = None) -> str:
        """
        Save cohort to JSON file.
        
        Args:
            cohort: List of child profiles
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"child_cohort_{len(cohort)}_{timestamp}.json"
        
        filepath = Path(__file__).parent.parent.parent / "data" / filename
        
        # Convert numpy types to ensure JSON serialization works
        json_safe_cohort = self._convert_numpy_types(cohort)
        
        with open(filepath, 'w') as f:
            json.dump(json_safe_cohort, f, indent=2)
        
        print(f"💾 Cohort saved to: {filepath}")
        return str(filepath)
    
    def load_cohort(self, filepath: str) -> List[Dict]:
        """Load cohort from JSON file."""
        with open(filepath, 'r') as f:
            cohort = json.load(f)
        
        print(f"📂 Loaded {len(cohort)} child profiles from {filepath}")
        return cohort


def main():
    """Demo function to test the child generator."""
    print("🧒 AI Early Childhood Education - Child Profile Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = ChildProfileGenerator(seed=42)  # For reproducible results
    
    # Generate a single child
    print("\n1. Generating a single child profile...")
    child = generator.generate_child()
    
    print(f"\nChild Profile:")
    print(f"Age: {child['age_years']} years ({child['age_months']} months)")
    print(f"Stage: {child['developmental_stage']}")
    print(f"Pace: {child['developmental_pace']}")
    print(f"Interests: {', '.join(child['interests'])}")
    print(f"Learning Style: {child['learning_style']}")
    print(f"Attention Span: {child['attention_span_minutes']} minutes")
    print(f"Top Skills: ", end="")
    
    # Show top 3 skills
    skills = child['skill_levels']
    top_skills = sorted(skills.items(), key=lambda x: x[1], reverse=True)[:3]
    print(", ".join([f"{skill}: {value:.1f}" for skill, value in top_skills]))
    
    # Generate a small cohort
    print(f"\n2. Generating a cohort of 20 children...")
    cohort = generator.generate_cohort(20)
    
    # Save cohort
    filepath = generator.save_cohort(cohort)
    
    # Show cohort statistics
    print(f"\nCohort Statistics:")
    stages = [child['developmental_stage'] for child in cohort]
    paces = [child['developmental_pace'] for child in cohort]
    
    print(f"Stages: {dict(zip(*np.unique(stages, return_counts=True)))}")
    print(f"Paces: {dict(zip(*np.unique(paces, return_counts=True)))}")
    
    print("\n✅ Child generator test complete!")


if __name__ == "__main__":
    main()
