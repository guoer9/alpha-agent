"""
进化个体定义
"""

import uuid
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class Individual:
    """进化个体 - 代表一个因子"""
    
    # 代码
    code: str
    
    # 标识
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    
    # 进化信息
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    # 评估结果
    fitness: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    reflection: str = ""
    
    # 状态
    is_evaluated: bool = False
    is_elite: bool = False
    error: str = ""
    
    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    evaluated_at: str = ""
    
    @property
    def code_hash(self) -> str:
        """代码哈希"""
        return hashlib.md5(self.code.encode()).hexdigest()[:8]
    
    def mark_evaluated(self, metrics: Dict, fitness: float, reflection: str = ""):
        """标记为已评估"""
        self.metrics = metrics
        self.fitness = fitness
        self.reflection = reflection
        self.is_evaluated = True
        self.evaluated_at = datetime.now().isoformat()
    
    def mark_failed(self, error: str):
        """标记为失败"""
        self.error = error
        self.fitness = 0.0
        self.is_evaluated = True
        self.evaluated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __repr__(self) -> str:
        status = "✓" if self.is_evaluated else "○"
        elite = "⭐" if self.is_elite else ""
        return f"Individual({self.id}, gen={self.generation}, fitness={self.fitness:.3f} {status}{elite})"


@dataclass
class GenerationStats:
    """单代统计"""
    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    min_fitness: float
    best_id: str
    valid_count: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvolutionHistory:
    """进化历史"""
    
    # 所有个体
    all_individuals: List[Individual] = field(default_factory=list)
    
    # 每代统计
    generation_stats: List[GenerationStats] = field(default_factory=list)
    
    # 最终精英
    final_elites: List[Individual] = field(default_factory=list)
    
    # 元信息
    task: str = ""
    config: Dict = field(default_factory=dict)
    start_time: str = ""
    end_time: str = ""
    total_generations: int = 0
    
    def add_generation(self, population: List[Individual], generation: int):
        """记录一代"""
        # 保存所有个体
        self.all_individuals.extend(population)
        
        # 计算统计
        fitnesses = [ind.fitness for ind in population]
        valid_count = sum(1 for f in fitnesses if f > 0)
        
        best_ind = max(population, key=lambda x: x.fitness)
        
        stats = GenerationStats(
            generation=generation,
            population_size=len(population),
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses) if fitnesses else 0,
            min_fitness=min(fitnesses),
            best_id=best_ind.id,
            valid_count=valid_count,
        )
        
        self.generation_stats.append(stats)
        self.total_generations = generation + 1
    
    def get_evolution_path(self, individual_id: str) -> List[Individual]:
        """追溯进化路径"""
        id_to_ind = {ind.id: ind for ind in self.all_individuals}
        
        path = []
        current_id = individual_id
        
        while current_id and current_id in id_to_ind:
            ind = id_to_ind[current_id]
            path.append(ind)
            current_id = ind.parent_ids[0] if ind.parent_ids else None
        
        return list(reversed(path))
    
    def get_best_of_generation(self, generation: int) -> Optional[Individual]:
        """获取某代最优个体"""
        gen_inds = [ind for ind in self.all_individuals if ind.generation == generation]
        if not gen_inds:
            return None
        return max(gen_inds, key=lambda x: x.fitness)
    
    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 50,
            "进化历史摘要",
            "=" * 50,
            f"任务: {self.task}",
            f"总代数: {self.total_generations}",
            f"总个体数: {len(self.all_individuals)}",
            "",
            "各代最优:",
        ]
        
        for stats in self.generation_stats:
            lines.append(
                f"  Gen {stats.generation}: "
                f"best={stats.best_fitness:.4f}, "
                f"avg={stats.avg_fitness:.4f}, "
                f"valid={stats.valid_count}/{stats.population_size}"
            )
        
        if self.final_elites:
            lines.append("")
            lines.append("最终精英:")
            for elite in self.final_elites[:3]:
                lines.append(f"  - {elite.id}: fitness={elite.fitness:.4f}")
        
        return "\n".join(lines)
