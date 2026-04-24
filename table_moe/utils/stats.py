class HybridStats:
    total_hits = 0
    total_queries = 0
    vision_hits = 0
    vision_total = 0
    text_hits = 0
    text_total = 0
    layer_stats = {}

    @classmethod
    def reset(cls):
        cls.total_hits = 0
        cls.total_queries = 0
        cls.vision_hits = 0
        cls.vision_total = 0
        cls.text_hits = 0
        cls.text_total = 0
        cls.layer_stats = {}

    @classmethod
    def update(cls, layer_id, hits, queries, vision_stats=None, text_stats=None):
        cls.total_hits += hits
        cls.total_queries += queries

        if vision_stats is not None:
            vision_hits, vision_total = vision_stats
            cls.vision_hits += vision_hits
            cls.vision_total += vision_total

        if text_stats is not None:
            text_hits, text_total = text_stats
            cls.text_hits += text_hits
            cls.text_total += text_total

        if layer_id not in cls.layer_stats:
            cls.layer_stats[layer_id] = {"hits": 0, "total": 0}
        cls.layer_stats[layer_id]["hits"] += hits
        cls.layer_stats[layer_id]["total"] += queries

    @classmethod
    def get_summary(cls):
        global_rate = (cls.total_hits / cls.total_queries * 100) if cls.total_queries > 0 else 0.0
        vision_rate = (cls.vision_hits / cls.vision_total * 100) if cls.vision_total > 0 else 0.0
        text_rate = (cls.text_hits / cls.text_total * 100) if cls.text_total > 0 else 0.0
        return (
            f"[Cache Stats] Global: {global_rate:.2f}% ({cls.total_hits}/{cls.total_queries}) | "
            f"Vision: {vision_rate:.2f}% | Text: {text_rate:.2f}%"
        )

    @classmethod
    def print_summary(cls):
        print(cls.get_summary())

    @classmethod
    def get_metrics_dict(cls):
        global_rate = (cls.total_hits / cls.total_queries) if cls.total_queries > 0 else 0.0
        vision_rate = (cls.vision_hits / cls.vision_total) if cls.vision_total > 0 else 0.0
        text_rate = (cls.text_hits / cls.text_total) if cls.text_total > 0 else 0.0

        return {
            "global": {"hits": cls.total_hits, "total": cls.total_queries, "rate": global_rate},
            "vision": {"hits": cls.vision_hits, "total": cls.vision_total, "rate": vision_rate},
            "text": {"hits": cls.text_hits, "total": cls.text_total, "rate": text_rate},
            "layer_stats": cls.layer_stats,
        }
