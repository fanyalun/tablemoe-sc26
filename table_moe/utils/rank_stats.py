from collections import defaultdict


class RankStats:
    _stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

    @classmethod
    def reset(cls):
        cls._stats = defaultdict(lambda: defaultdict(lambda: {"hits": 0, "total": 0}))

    @classmethod
    def update(cls, layer_id, hits_per_rank, total_tokens):
        hits_list = hits_per_rank.tolist()
        for rank, hits in enumerate(hits_list):
            cls._stats[layer_id][rank]["hits"] += int(hits)
            cls._stats[layer_id][rank]["total"] += int(total_tokens)

    @classmethod
    def get_layer_stats(cls):
        results = {}
        for layer_id, ranks in cls._stats.items():
            results[layer_id] = {}
            for rank, data in ranks.items():
                total = data["total"]
                results[layer_id][rank] = data["hits"] / total if total > 0 else 0.0
        return results

    @classmethod
    def get_average_stats(cls):
        rank_totals = defaultdict(lambda: {"hits": 0, "total": 0})
        for ranks in cls._stats.values():
            for rank, data in ranks.items():
                rank_totals[rank]["hits"] += data["hits"]
                rank_totals[rank]["total"] += data["total"]

        avg_stats = {}
        for rank, data in rank_totals.items():
            avg_stats[rank] = data["hits"] / data["total"] if data["total"] > 0 else 0.0
        return avg_stats
