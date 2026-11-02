import pandas as pd
import numpy as np

def generate_training_data(num_records=10000):
    np.random.seed(42)
    data = []
    
    for _ in range(num_records):
        # 1. Simulate the input features
        hour = np.random.randint(0, 24)
        is_peak_hour = 1 if 9 <= hour <= 17 else 0
        category_hit_rate = np.random.uniform(0.1, 0.9)
        user_priority = 1 if np.random.rand() < 0.1 else 0
        
        # 2. Simulate the target (Ground Truth): Will it be accessed again soon?
        # We artificially increase the probability for priority files and peak hours
        probability_of_reuse = 0.2  # Base chance
        probability_of_reuse += (is_peak_hour * 0.25)
        probability_of_reuse += (category_hit_rate * 0.20)
        probability_of_reuse += (user_priority * 0.30)
        
        # Cap probability at 0.95
        probability_of_reuse = min(probability_of_reuse, 0.95)
        
        # Determine actual reuse based on the calculated probability
        will_be_reused = 1 if np.random.rand() < probability_of_reuse else 0
        
        data.append({
            'is_peak_hour': is_peak_hour,
            'category_hit_rate': category_hit_rate,
            'user_priority': user_priority,
            'will_be_reused': will_be_reused
        })
        
    df = pd.DataFrame(data)
    df.to_csv("cache_training_data.csv", index=False)
    print(f"Successfully generated {num_records} records in 'cache_training_data.csv'")

if __name__ == "__main__":
    generate_training_data()