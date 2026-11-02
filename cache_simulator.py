import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from sklearn.tree import DecisionTreeClassifier

# =====================================================================
# 1. ML PREDICTOR
# =====================================================================
class FastMLPredictor:
    def __init__(self, max_depth=4):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.is_trained = False

    def train_and_compile(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        self.is_trained = True

    def predict_is_valid(self, features):
        if not self.is_trained: return True
        return bool(self.clf.predict([features])[0])


# =====================================================================
# 2. DYNAMIC OFFLINE TRAINER
# =====================================================================
def profile_trace_for_training(trace_data, lookahead_window=200):
    X_nc, X_ctx, y = [], [], []
    tracker = {}
    cat_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    
    future_accesses = {}
    next_access_list = [float('inf')] * len(trace_data)
    for i in range(len(trace_data)-1, -1, -1):
        f = trace_data[i]['file']
        next_access_list[i] = future_accesses.get(f, float('inf'))
        future_accesses[f] = i

    for i, req in enumerate(trace_data):
        f = req['file']
        cat = req['category']
        hour = req['hour']
        prio = req['user_priority']
        size = req.get('file_size_mb', 1)
        
        cat_stats[cat]['total'] += 1
        is_peak = 1 if (cat == 'Development' and 9 <= hour <= 17) else 0
        cat_rate = (cat_stats[cat]['hits'] / cat_stats[cat]['total'])
        
        if f not in tracker:
            tracker[f] = {'hits': 0, 'last_io': i, 'inserted': i}
        else:
            tracker[f]['hits'] += 1
            tracker[f]['last_io'] = i
            cat_stats[cat]['hits'] += 1
            
        hits = tracker[f]['hits']
        recency = i - tracker[f]['last_io']
        ttl = i - tracker[f]['inserted']
        lifetime_proxy = min(len(tracker), 100)
        
        label = 1 if (next_access_list[i] - i) <= lookahead_window else 0
        
        X_nc.append([hits, recency, size, ttl, lifetime_proxy])
        X_ctx.append([hits, recency, size, ttl, is_peak, cat_rate, prio])
        y.append(label)
        
    return X_nc, X_ctx, y


# =====================================================================
# 3. BASELINE CACHES (LRU, LFU, MARKOV)
# =====================================================================
class BaseCache:
    def __init__(self, max_size_mb, compute_overhead_ms):
        self.max_size_mb = max_size_mb
        self.current_size = 0
        self.compute_overhead_ms = compute_overhead_ms
        self.total_hits = 0
        self.total_misses = 0
        self.total_latency_ms = 0.0

    @property
    def hit_ratio(self):
        total = self.total_hits + self.total_misses
        return (self.total_hits / total * 100) if total > 0 else 0.0
    @property
    def avg_latency(self):
        total = self.total_hits + self.total_misses
        return (self.total_latency_ms / total) if total > 0 else 0.0

class LRUCache(BaseCache):
    def __init__(self, max_size_mb):
        super().__init__(max_size_mb, compute_overhead_ms=0.001)
        self.cache = OrderedDict()

    def access(self, req):
        lpa, size = req['file'], req.get('file_size_mb', 1)
        self.total_latency_ms += self.compute_overhead_ms
        if size > self.max_size_mb:
            self.total_misses += 1; self.total_latency_ms += 10.0; return

        if lpa in self.cache:
            self.total_hits += 1; self.total_latency_ms += 0.05
            self.cache.move_to_end(lpa)
        else:
            self.total_misses += 1; self.total_latency_ms += 10.0
            while self.current_size + size > self.max_size_mb:
                evicted, e_size = self.cache.popitem(last=False)
                self.current_size -= e_size
            self.cache[lpa] = size
            self.current_size += size

class LFUCache(BaseCache):
    def __init__(self, max_size_mb):
        super().__init__(max_size_mb, compute_overhead_ms=0.002)
        self.cache = {}
        self.freq = defaultdict(int)

    def access(self, req):
        lpa, size = req['file'], req.get('file_size_mb', 1)
        self.total_latency_ms += self.compute_overhead_ms
        if size > self.max_size_mb:
            self.total_misses += 1; self.total_latency_ms += 10.0; return

        if lpa in self.cache:
            self.total_hits += 1; self.total_latency_ms += 0.05
            self.freq[lpa] += 1
        else:
            self.total_misses += 1; self.total_latency_ms += 10.0
            while self.current_size + size > self.max_size_mb:
                lfu_key = min(self.cache.keys(), key=lambda k: self.freq[k])
                e_size = self.cache.pop(lfu_key)
                del self.freq[lfu_key]
                self.current_size -= e_size
            self.cache[lpa] = size
            self.freq[lpa] = 1
            self.current_size += size

class MarkovLRUCache(BaseCache):
    def __init__(self, max_size_mb):
        super().__init__(max_size_mb, compute_overhead_ms=0.003)
        self.cache = OrderedDict()
        self.transition_matrix = {}
        self.known_sizes = {}
        self.last_lpa = None

    def access(self, req):
        lpa, size = req['file'], req.get('file_size_mb', 1)
        self.total_latency_ms += self.compute_overhead_ms
        self.known_sizes[lpa] = size
        
        if size > self.max_size_mb:
            self.total_misses += 1; self.total_latency_ms += 10.0; return

        if self.last_lpa: self.transition_matrix[self.last_lpa] = lpa

        if lpa in self.cache:
            self.total_hits += 1; self.total_latency_ms += 0.05
            self.cache.move_to_end(lpa)
        else:
            self.total_misses += 1; self.total_latency_ms += 10.0
            while self.current_size + size > self.max_size_mb:
                evicted, e_size = self.cache.popitem(last=False)
                self.current_size -= e_size
            self.cache[lpa] = size
            self.current_size += size

        # Markov Prefetch
        if lpa in self.transition_matrix:
            next_lpa = self.transition_matrix[lpa]
            next_size = self.known_sizes.get(next_lpa, 1)
            if next_lpa not in self.cache and (self.current_size + next_size <= self.max_size_mb):
                self.cache[next_lpa] = next_size
                self.current_size += next_size

        self.last_lpa = lpa


# =====================================================================
# 4. ML MODELS (BASE PAPER & PROPOSED)
# =====================================================================
class CacheBlockML:
    def __init__(self, lpa, size, category=None, user_priority=None):
        self.lpa = lpa; self.size = size
        self.category = category; self.user_priority = user_priority
        self.hits = 0; self.last_accessed_io = 0; self.inserted_io = 0            

class NCacheBase(BaseCache):
    def __init__(self, max_size_mb, ml_predictor, window_size=10):
        super().__init__(max_size_mb, compute_overhead_ms=0.005)
        self.ml, self.window_size, self.global_io_counter = ml_predictor, window_size, 0     
        self.link_l1, self.link_l2 = OrderedDict(), OrderedDict() 

    def access(self, req):
        lpa, size = req['file'], req.get('file_size_mb', 1)
        self.total_latency_ms += self.compute_overhead_ms
        if size > self.max_size_mb:
            self.total_misses += 1; self.total_latency_ms += 10.0; return

        self.global_io_counter += 1
        if lpa in self.link_l1:
            self.total_hits += 1; self.total_latency_ms += 0.05
            block = self.link_l1[lpa]
            block.hits += 1; block.last_accessed_io = self.global_io_counter
            self.link_l1.move_to_end(lpa, last=True)
            if lpa in self.link_l2: del self.link_l2[lpa]
        else:
            self.total_misses += 1; self.total_latency_ms += 10.0
            while self.current_size + size > self.max_size_mb:
                self._evict()
            
            new_block = CacheBlockML(lpa, size)
            new_block.inserted_io = new_block.last_accessed_io = self.global_io_counter
            self.link_l1[lpa] = new_block
            self.current_size += size
            
            features = [new_block.hits, 0, size, 0, len(self.link_l1)]
            if not self.ml.predict_is_valid(features): self.link_l2[lpa] = True 

    def _evict(self):
        evicted_lpa = None
        for lpa in list(self.link_l1.keys())[:self.window_size]:
            if lpa in self.link_l2: evicted_lpa = lpa; break
        if evicted_lpa is None: evicted_lpa = next(iter(self.link_l1))
        self.current_size -= self.link_l1[evicted_lpa].size
        del self.link_l1[evicted_lpa]
        if evicted_lpa in self.link_l2: del self.link_l2[evicted_lpa]

class ContextAwareCache(BaseCache):
    def __init__(self, max_size_mb, ml_predictor, window_size=10):
        super().__init__(max_size_mb, compute_overhead_ms=0.005)
        self.ml, self.window_size, self.global_io_counter = ml_predictor, window_size, 0     
        self.link_l1, self.link_l2 = OrderedDict(), OrderedDict() 
        self.cat_stats, self.known_meta, self.transition_matrix = defaultdict(lambda: {"hits": 0, "total": 0}), {}, {}
        self.last_lpa = None

    def access(self, req):
        lpa, hour, cat, prio, size = req['file'], req['hour'], req['category'], req['user_priority'], req.get('file_size_mb', 1)
        self.total_latency_ms += self.compute_overhead_ms
        if size > self.max_size_mb:
            self.total_misses += 1; self.total_latency_ms += 10.0; return

        self.global_io_counter += 1
        self.cat_stats[cat]['total'] += 1
        self.known_meta[lpa] = {'category': cat, 'user_priority': prio, 'size': size}
        if self.last_lpa: self.transition_matrix[self.last_lpa] = lpa

        if lpa in self.link_l1:
            self.total_hits += 1; self.total_latency_ms += 0.05
            self.cat_stats[cat]['hits'] += 1
            block = self.link_l1[lpa]
            block.hits += 1; block.last_accessed_io = self.global_io_counter
            self.link_l1.move_to_end(lpa, last=True)
            if lpa in self.link_l2: del self.link_l2[lpa]
        else:
            self.total_misses += 1; self.total_latency_ms += 10.0
            while self.current_size + size > self.max_size_mb: self._evict()
            
            new_block = CacheBlockML(lpa, size, cat, prio)
            new_block.inserted_io = new_block.last_accessed_io = self.global_io_counter
            self.link_l1[lpa] = new_block
            self.current_size += size
            
            features = self._extract_features(new_block, hour)
            if not self.ml.predict_is_valid(features): self.link_l2[lpa] = True 

        # Context-Aware Prefetch (Confidence-Based)
        if lpa in self.transition_matrix:
            next_lpa = self.transition_matrix[lpa]
            if next_lpa not in self.link_l1:
                meta = self.known_meta.get(next_lpa)
                if meta and (self.current_size + meta['size'] <= self.max_size_mb):
                    
                    # 1. Create a temporary block to test the ML features BEFORE committing
                    temp_pb = CacheBlockML(next_lpa, meta['size'], meta['category'], meta['user_priority'])
                    temp_pb.inserted_io = temp_pb.last_accessed_io = self.global_io_counter
                    
                    p_feats = self._extract_features(temp_pb, hour)
                    
                    # 2. ONLY prefetch if the ML model is confident it is valid (Label 1)
                    if self.ml.predict_is_valid(p_feats):
                        self.link_l1[next_lpa] = temp_pb
                        self.current_size += meta['size']
                    # If invalid (Label 0), we reject the prefetch entirely to prevent cache pollution.

        self.last_lpa = lpa

    def _extract_features(self, b, hour):
        is_peak = 1 if (b.category == 'Development' and 9 <= hour <= 17) else 0
        cat_rate = (self.cat_stats[b.category]['hits'] / self.cat_stats[b.category]['total']) if self.cat_stats[b.category]['total'] > 0 else 0.5
        return [b.hits, self.global_io_counter - b.last_accessed_io, b.size, self.global_io_counter - b.inserted_io, is_peak, cat_rate, b.user_priority]

    def _evict(self):
        evicted_lpa = None
        for lpa in list(self.link_l1.keys())[:self.window_size]:
            if lpa in self.link_l2: evicted_lpa = lpa; break
        if evicted_lpa is None: evicted_lpa = next(iter(self.link_l1))
        self.current_size -= self.link_l1[evicted_lpa].size
        del self.link_l1[evicted_lpa]
        if evicted_lpa in self.link_l2: del self.link_l2[evicted_lpa]


# =====================================================================
# 5. BENCHMARK RUNNER & PLOTTER
# =====================================================================
if __name__ == "__main__":
    print("Loading access_trace.csv...")
    try:
        trace_data = pd.read_csv('access_trace.csv').to_dict('records')
    except FileNotFoundError:
        print("Error: access_trace.csv not found.")
        exit()

    print("Profiling trace to generate dynamic training data...")
    X_ncache, X_context, y_labels = profile_trace_for_training(trace_data)
    
    ncache_ml = FastMLPredictor(max_depth=4)
    ncache_ml.train_and_compile(X_ncache, y_labels)

    context_ml = FastMLPredictor(max_depth=4)
    context_ml.train_and_compile(X_context, y_labels)
    print("Models trained successfully!\n")

    test_sizes_mb = [16, 32, 64]
    results = {
        "LRU": {"hr": [], "lat": [], "color": "gray", "marker": "o"},
        "LFU": {"hr": [], "lat": [], "color": "cyan", "marker": "s"},
        "Markov": {"hr": [], "lat": [], "color": "orange", "marker": "d"},
        "NCache (Base)": {"hr": [], "lat": [], "color": "blue", "marker": "x"},
        "Context (Proposed)": {"hr": [], "lat": [], "color": "green", "marker": "^"}
    }

    print("="*95)
    print(f"{'FULL EVALUATION METRICS (BASELINES VS PROPOSED)':^95}")
    print("="*95)

    for cap in test_sizes_mb:
        policies = {
            "LRU": LRUCache(cap),
            "LFU": LFUCache(cap),
            "Markov": MarkovLRUCache(cap),
            "NCache (Base)": NCacheBase(cap, ncache_ml),
            "Context (Proposed)": ContextAwareCache(cap, context_ml)
        }
        
        for req in trace_data:
            for name, policy in policies.items():
                policy.access(req)
                
        for name, policy in policies.items():
            results[name]["hr"].append(policy.hit_ratio)
            results[name]["lat"].append(policy.avg_latency)

        print(f"\nCAPACITY LIMIT: {cap} MB")
        print("-" * 95)
        print(f"{'Policy':<25} | {'Hits':<10} | {'Misses':<10} | {'Hit Ratio':<12} | {'Avg Latency':<12}")
        print("-" * 95)
        for name, p in policies.items():
            print(f"{name:<25} | {p.total_hits:<10} | {p.total_misses:<10} | {p.hit_ratio:>6.2f}%      | {p.avg_latency:>6.3f} ms")
        print("-" * 95)

    # --- GENERATE PLOTS ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(test_sizes_mb, data["hr"], marker=data["marker"], linestyle='--', linewidth=(2.5 if "Proposed" in name else 1.5), color=data["color"], label=name)
    plt.title('Hit Ratio vs. Cache Capacity')
    plt.xlabel('Max Cache Size (MB)')
    plt.ylabel('Hit Ratio (%)')
    plt.xticks(test_sizes_mb)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(test_sizes_mb, data["lat"], marker=data["marker"], linestyle='--', linewidth=(2.5 if "Proposed" in name else 1.5), color=data["color"], label=name)
    plt.title('Avg Latency vs. Cache Capacity')
    plt.xlabel('Max Cache Size (MB)')
    plt.ylabel('Average Latency (ms)')
    plt.xticks(test_sizes_mb)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('full_comparative_results.png', dpi=300)
    print("\nSimulation complete! Chart saved as 'full_comparative_results.png'")