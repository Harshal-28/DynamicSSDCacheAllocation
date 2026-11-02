import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. ML PREDICTOR
# =====================================================================
class FastMLPredictor:
    def __init__(self, max_depth=4):
        self.clf = DecisionTreeClassifier(max_depth=max_depth)
        self.is_trained = False

    def train_and_compile(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        self.is_trained = True

    def predict_is_valid(self, features):
        if not self.is_trained: return True
        return bool(self.clf.predict([features])[0])


# =====================================================================
# 2. BASE PAPER: NCACHE (HARDWARE-LEVEL ONLY)
# =====================================================================
#/mnt/c/Users/Harshal/Desktop/OS-Paper/Project
class CacheBlockNCache:
    def __init__(self, lpa, size):
        self.lpa = lpa                  
        self.size = size
        self.hits = 0                   # Frequency
        self.last_accessed_io = 0       # Used for Recency[cite: 1]
        self.inserted_io = 0            # Used for Timetolive[cite: 1]

class NCacheBase:
    """Simulates the NCache paper using only hardware-level features[cite: 1]."""
    def __init__(self, max_size_mb, ml_predictor, window_size=10):
        self.max_size_mb = max_size_mb
        self.current_size = 0
        self.ml = ml_predictor
        self.window_size = window_size  
        self.global_io_counter = 0     
        
        self.total_hits = 0
        self.total_misses = 0
        self.total_latency_ms = 0.0

        # Dual Linked List Architecture[cite: 1]
        self.link_l1 = OrderedDict() 
        self.link_l2 = OrderedDict() 

    @property
    def hit_ratio(self):
        total = self.total_hits + self.total_misses
        return (self.total_hits / total * 100) if total > 0 else 0.0
    
    @property
    def avg_latency(self):
        total = self.total_hits + self.total_misses
        return (self.total_latency_ms / total) if total > 0 else 0.0

    def _extract_features(self, block, current_position_from_tail):
        # Strictly the 5 features from the base paper[cite: 1]
        recency = self.global_io_counter - block.last_accessed_io
        ttl = self.global_io_counter - block.inserted_io
        return [block.hits, recency, block.size, ttl, current_position_from_tail]

    def access(self, req):
        lpa = req['file']
        size = req.get('file_size_mb', 1)
        self.total_latency_ms += 0.005 # ML overhead

        if size > self.max_size_mb:
            self.total_misses += 1
            self.total_latency_ms += 10.0
            return

        self.global_io_counter += 1

        if lpa in self.link_l1:
            self.total_hits += 1
            self.total_latency_ms += 0.05
            block = self.link_l1[lpa]
            block.hits += 1
            block.last_accessed_io = self.global_io_counter
            
            self.link_l1.move_to_end(lpa, last=True)
            if lpa in self.link_l2: del self.link_l2[lpa]
        else:
            self.total_misses += 1
            self.total_latency_ms += 10.0
            
            while self.current_size + size > self.max_size_mb:
                self._evict()
            
            new_block = CacheBlockNCache(lpa, size)
            new_block.inserted_io = self.global_io_counter
            new_block.last_accessed_io = self.global_io_counter
            
            self.link_l1[lpa] = new_block
            self.current_size += size
            
            pos_from_tail = len(self.link_l1)
            features = self._extract_features(new_block, pos_from_tail)
            if not self.ml.predict_is_valid(features):
                self.link_l2[lpa] = True 

    def _evict(self):
        evicted_lpa = None
        tail_candidates = list(self.link_l1.keys())[:self.window_size]
        for lpa in tail_candidates:
            if lpa in self.link_l2:
                evicted_lpa = lpa
                break
        
        if evicted_lpa is None:
            evicted_lpa = next(iter(self.link_l1))

        self.current_size -= self.link_l1[evicted_lpa].size
        del self.link_l1[evicted_lpa]
        if evicted_lpa in self.link_l2: del self.link_l2[evicted_lpa]


# =====================================================================
# 3. YOUR PROPOSED PAPER: CONTEXT-AWARE INLINE CACHE
# =====================================================================
class CacheBlockContext:
    def __init__(self, lpa, size, category, user_priority):
        self.lpa = lpa                  
        self.size = size
        self.category = category
        self.user_priority = user_priority
        self.hits = 0                   
        self.last_accessed_io = 0       
        self.inserted_io = 0            

class ContextAwareCache:
    """Your proposed model bridging the semantic gap and adding prefetching."""
    def __init__(self, max_size_mb, ml_predictor, window_size=10):
        self.max_size_mb = max_size_mb
        self.current_size = 0
        self.ml = ml_predictor
        self.window_size = window_size  
        self.global_io_counter = 0     
        
        self.total_hits = 0
        self.total_misses = 0
        self.total_latency_ms = 0.0

        self.link_l1 = OrderedDict() 
        self.link_l2 = OrderedDict() 
        
        self.category_stats = defaultdict(lambda: {"hits": 0, "total": 0})
        self.known_metadata = {}     
        self.transition_matrix = {}  
        self.last_accessed_lpa = None

    @property
    def hit_ratio(self):
        total = self.total_hits + self.total_misses
        return (self.total_hits / total * 100) if total > 0 else 0.0
        
    @property
    def avg_latency(self):
        total = self.total_hits + self.total_misses
        return (self.total_latency_ms / total) if total > 0 else 0.0

    def _extract_features(self, block, current_hour):
        is_peak = 1 if (block.category == 'Development' and 9 <= current_hour <= 17) else 0
        stats = self.category_stats[block.category]
        cat_rate = (stats['hits'] / stats['total']) if stats['total'] > 0 else 0.5
        ttl = self.global_io_counter - block.inserted_io
        recency = self.global_io_counter - block.last_accessed_io
        
        return [block.hits, recency, block.size, ttl, is_peak, cat_rate, block.user_priority]

    def access(self, req):
        lpa = req['file']
        hour = req['hour']
        cat = req['category']
        user_prio = req['user_priority']
        size = req.get('file_size_mb', 1)
        self.total_latency_ms += 0.005 # ML overhead

        if size > self.max_size_mb:
            self.total_misses += 1
            self.total_latency_ms += 10.0
            return

        self.global_io_counter += 1
        self.category_stats[cat]['total'] += 1
        self.known_metadata[lpa] = {'category': cat, 'user_priority': user_prio, 'size': size}

        if self.last_accessed_lpa:
            self.transition_matrix[self.last_accessed_lpa] = lpa

        if lpa in self.link_l1:
            self.total_hits += 1
            self.total_latency_ms += 0.05
            self.category_stats[cat]['hits'] += 1
            
            block = self.link_l1[lpa]
            block.hits += 1
            block.last_accessed_io = self.global_io_counter
            
            self.link_l1.move_to_end(lpa, last=True)
            if lpa in self.link_l2: del self.link_l2[lpa]
        else:
            self.total_misses += 1
            self.total_latency_ms += 10.0
            
            while self.current_size + size > self.max_size_mb:
                self._evict()
            
            new_block = CacheBlockContext(lpa, size, cat, user_prio)
            new_block.inserted_io = self.global_io_counter
            new_block.last_accessed_io = self.global_io_counter
            
            self.link_l1[lpa] = new_block
            self.current_size += size
            
            features = self._extract_features(new_block, hour)
            if not self.ml.predict_is_valid(features):
                self.link_l2[lpa] = True 

        # Proactive Sequential Prefetching (Not in base paper)
        if lpa in self.transition_matrix:
            next_lpa = self.transition_matrix[lpa]
            if next_lpa not in self.link_l1:
                meta = self.known_metadata.get(next_lpa)
                if meta and (self.current_size + meta['size'] <= self.max_size_mb):
                    pb = CacheBlockContext(next_lpa, meta['size'], meta['category'], meta['user_priority'])
                    pb.inserted_io = self.global_io_counter
                    pb.last_accessed_io = self.global_io_counter
                    
                    self.link_l1[next_lpa] = pb
                    self.current_size += meta['size']
                    
                    p_feats = self._extract_features(pb, hour)
                    if not self.ml.predict_is_valid(p_feats):
                        self.link_l2[next_lpa] = True 

        self.last_accessed_lpa = lpa

    def _evict(self):
        evicted_lpa = None
        tail_candidates = list(self.link_l1.keys())[:self.window_size]
        for lpa in tail_candidates:
            if lpa in self.link_l2:
                evicted_lpa = lpa
                break
        if evicted_lpa is None:
            evicted_lpa = next(iter(self.link_l1))

        self.current_size -= self.link_l1[evicted_lpa].size
        del self.link_l1[evicted_lpa]
        if evicted_lpa in self.link_l2: del self.link_l2[evicted_lpa]


# =====================================================================
# 4. BENCHMARK RUNNER & PLOTTER
# =====================================================================
if __name__ == "__main__":
    # --- Train NCache Model (Hardware features only) ---
    ncache_ml = FastMLPredictor(max_depth=4)
    # [hits, recency, size, ttl, lifetime_idx] -> Valid(1) / Invalid(0)
    X_ncache = [[0, 50, 4, 100, 1], [3, 5, 1, 10, 5], [0, 200, 8, 250, 2], [5, 2, 2, 20, 8]]
    y_ncache = [0, 1, 0, 1]
    ncache_ml.train_and_compile(X_ncache, y_ncache)

    # --- Train Context Model (Semantic features added) ---
    context_ml = FastMLPredictor(max_depth=4)
    # [hits, recency, size, ttl, is_peak, cat_rate, user_prio] -> Valid(1) / Invalid(0)
    X_context = [[0, 50, 4, 100, 0, 0.1, 0.0], [3, 5, 1, 10, 1, 0.8, 1.0], [0, 200, 8, 250, 0, 0.2, 0.0], [5, 2, 2, 20, 1, 0.9, 1.0]]
    y_context = [0, 1, 0, 1]
    context_ml.train_and_compile(X_context, y_context)

    # --- Load Dataset ---
    print("Loading access_trace.csv...")
    try:
        trace_data = pd.read_csv('access_trace.csv').to_dict('records')
    except FileNotFoundError:
        print("Error: access_trace.csv not found.")
        exit()

    test_sizes_mb = [16, 32, 64]
    
    ncache_hr, ncache_lat = [], []
    context_hr, context_lat = [], []

    print("\nRunning Comparative Simulation...")
    for cap in test_sizes_mb:
        print(f"Testing Capacity: {cap} MB")
        
        nc = NCacheBase(max_size_mb=cap, ml_predictor=ncache_ml, window_size=10)
        cc = ContextAwareCache(max_size_mb=cap, ml_predictor=context_ml, window_size=10)
        
        for req in trace_data:
            nc.access(req)
            cc.access(req)
            
        ncache_hr.append(nc.hit_ratio)
        ncache_lat.append(nc.avg_latency)
        
        context_hr.append(cc.hit_ratio)
        context_lat.append(cc.avg_latency)

    # --- Generate Plots ---
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Hit Ratio
    plt.subplot(1, 2, 1)
    plt.plot(test_sizes_mb, ncache_hr, marker='s', linestyle='--', color='blue', label='NCache (Base Paper)')
    plt.plot(test_sizes_mb, context_hr, marker='^', linewidth=2.5, color='green', label='Context-Aware (Proposed)')
    plt.title('Hit Ratio vs. Cache Capacity')
    plt.xlabel('Max Cache Size (MB)')
    plt.ylabel('Hit Ratio (%)')
    plt.xticks(test_sizes_mb)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # Subplot 2: Average Latency
    plt.subplot(1, 2, 2)
    plt.plot(test_sizes_mb, ncache_lat, marker='s', linestyle='--', color='blue', label='NCache (Base Paper)')
    plt.plot(test_sizes_mb, context_lat, marker='^', linewidth=2.5, color='green', label='Context-Aware (Proposed)')
    plt.title('Avg Latency vs. Cache Capacity')
    plt.xlabel('Max Cache Size (MB)')
    plt.ylabel('Average Latency (ms)')
    plt.xticks(test_sizes_mb)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('comparative_results.png', dpi=300)
    print("\nSimulation complete! Chart saved as 'comparative_results.png'")