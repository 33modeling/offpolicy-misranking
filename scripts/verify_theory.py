#!/usr/bin/env python3
"""Executable checks for the counterexamples and geometric lemmas in concept.md."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from itertools import product

Trajectory = tuple[float, float, float]


def trajectories(example: str, policy: str, epsilon: float) -> list[Trajectory]:
    """Return (probability, reward, relevant score) for each two-token path."""
    paths: list[Trajectory] = []
    for first, second in product((0, 1), repeat=2):
        if example == "prefix_only":
            first_prob = 0.5
            behavior_success = 0.5 + epsilon if first else 0.5 - epsilon
            success_prob = (
                behavior_success if policy == "behavior" else 1.0 - behavior_success
            )
            probability = first_prob * (success_prob if second else 1.0 - success_prob)
            reward = float(second)
            score = first - 0.5
        elif example == "future_only":
            behavior_first_zero = 0.5 + epsilon
            first_zero_prob = (
                behavior_first_zero
                if policy == "behavior"
                else 1.0 - behavior_first_zero
            )
            first_prob = first_zero_prob if first == 0 else 1.0 - first_zero_prob
            probability = first_prob * 0.5
            reward = float(second if first == 0 else 1 - second)
            score = second - 0.5
        else:
            raise ValueError(f"unknown example: {example}")
        paths.append((probability, reward, score))

    assert math.isclose(sum(path[0] for path in paths), 1.0, abs_tol=1e-12)
    return paths


def raw_gradient(paths: Iterable[Trajectory]) -> float:
    return sum(probability * reward * score for probability, reward, score in paths)


def expected_standardized_group_gradient(paths: list[Trajectory], group_size: int) -> float:
    """Exactly enumerate the binary group-normalized gradient expectation."""
    expectation = 0.0
    for indices in product(range(len(paths)), repeat=group_size):
        group = [paths[index] for index in indices]
        group_probability = math.prod(path[0] for path in group)
        mean_reward = sum(path[1] for path in group) / group_size
        variance = sum((path[1] - mean_reward) ** 2 for path in group) / group_size
        if variance == 0.0:
            continue
        std = math.sqrt(variance)
        gradient = sum(
            ((reward - mean_reward) / std) * score for _, reward, score in group
        ) / group_size
        expectation += group_probability * gradient
    return expectation


def trajectory_kl(example: str, epsilon: float) -> float:
    """KL(π‖β) on the two-token path measure."""
    current = trajectories(example, "current", epsilon)
    behavior = trajectories(example, "behavior", epsilon)
    kl = 0.0
    for (p_pi, _, _), (p_b, _, _) in zip(current, behavior):
        if p_pi > 0.0:
            kl += p_pi * math.log(p_pi / max(p_b, 1e-15))
    return kl


def is_cells(example: str, epsilon: float) -> dict[str, float]:
    """Population g00/g10/g01/g11 and g_π from the four paths (scalar score)."""
    behavior = trajectories(example, "behavior", epsilon)
    current = trajectories(example, "current", epsilon)
    cells = {key: 0.0 for key in ("pi", "00", "10", "01", "11")}
    for (p_b, reward, score), (p_pi, _, _) in zip(behavior, current):
        ratio = p_pi / p_b
        cells["pi"] += p_pi * reward * score
        # One-step families: prefix_only has r1=1, r2=π/β on suffix;
        # future_only has r2=1, r1=π/β on first action. For these MDPs the
        # scalar score lives on one token, so the four cells collapse as in
        # concept.md §4. Reconstruct via path-level ratios.
        cells["11"] += p_b * ratio * reward * score
        if example == "prefix_only":
            # occupancy identical ⇒ P_t=1; continuation differs ⇒ S_t = r2.
            cells["00"] += p_b * reward * score  # token-ratio on scored token ≈ 1
            cells["10"] += p_b * reward * score  # prefix-corrected, Q^β
            cells["01"] += p_b * ratio * reward * score
        else:
            # continuation identical ⇒ S_t=1; occupancy differs.
            cells["00"] += p_b * reward * score
            cells["01"] += p_b * reward * score
            cells["10"] += p_b * ratio * reward * score
    return cells


def check_sign_reversals() -> None:
    for epsilon in (0.4, 0.1, 0.01, 0.001):
        for example in ("prefix_only", "future_only"):
            behavior = trajectories(example, "behavior", epsilon)
            current = trajectories(example, "current", epsilon)
            behavior_raw = raw_gradient(behavior)
            current_raw = raw_gradient(current)
            assert math.isclose(behavior_raw, epsilon / 2.0, abs_tol=1e-12)
            assert math.isclose(current_raw, -epsilon / 2.0, abs_tol=1e-12)

            cells = is_cells(example, epsilon)
            assert math.isclose(cells["pi"], current_raw, abs_tol=1e-12)
            assert math.isclose(cells["11"], current_raw, abs_tol=1e-9)
            # Relative error of the relevant one-sided cell is exactly -1.
            if example == "prefix_only":
                assert math.isclose(cells["10"], -cells["pi"], abs_tol=1e-12)
                assert math.isclose(cells["00"], cells["10"], abs_tol=1e-12)
            else:
                assert math.isclose(cells["01"], -cells["pi"], abs_tol=1e-12)
                assert math.isclose(cells["00"], cells["01"], abs_tol=1e-12)

            kl = trajectory_kl(example, epsilon)
            assert kl >= 0.0
            if epsilon <= 0.1:
                # KL(π‖β)=O(ε²): ratio KL/ε² stays bounded.
                assert kl / (epsilon**2) < 20.0

            for group_size in (2, 4, 8):
                behavior_group = expected_standardized_group_gradient(
                    behavior, group_size
                )
                current_group = expected_standardized_group_gradient(current, group_size)
                assert behavior_group > 0.0
                assert current_group < 0.0
                assert math.isclose(
                    behavior_group, -current_group, rel_tol=1e-10, abs_tol=1e-12
                )


def check_ranking_reversal() -> None:
    """Two prompts, shared validation direction v = e_1: true top-1 flips."""
    epsilon = 0.1
    plus = is_cells("prefix_only", epsilon)
    # Prompt x+: true g = -ε/2 under naming of §4, which is the "current" cell.
    # Alignment with v=+1: true score = g_π, one-sided = g_10.
    true_plus = plus["pi"]
    sided_plus = plus["10"]
    true_minus = -true_plus
    sided_minus = -sided_plus
    assert true_plus < 0.0 < true_minus
    # If v points with the current policy of x- (positive), true ranking is
    # x- > x+, one-sided ranking is x+ > x-.
    assert true_minus > true_plus
    assert sided_plus > sided_minus
    assert math.isclose(sided_plus, -true_plus, abs_tol=1e-12)


def expected_loo_gradient(success_probability: float, group_size: int) -> float:
    expectation = 0.0
    for rewards in product((0, 1), repeat=group_size):
        successes = sum(rewards)
        probability = success_probability**successes * (
            1.0 - success_probability
        ) ** (group_size - successes)
        terms = []
        for reward in rewards:
            other_mean = (successes - reward) / (group_size - 1)
            score = reward - success_probability
            terms.append((reward - other_mean) * score)
        expectation += probability * sum(terms) / group_size
    return expectation


def check_leave_one_out_unbiasedness() -> None:
    for success_probability in (0.1, 0.3, 0.5, 0.9):
        target = success_probability * (1.0 - success_probability)
        for group_size in (2, 4, 8):
            estimate = expected_loo_gradient(success_probability, group_size)
            assert math.isclose(estimate, target, rel_tol=1e-12, abs_tol=1e-12)


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def angle(left: list[float], right: list[float]) -> float:
    cosine = sum(a * b for a, b in zip(left, right)) / (norm(left) * norm(right))
    return math.acos(max(-1.0, min(1.0, cosine)))


def check_angular_radius() -> None:
    center_norm = 3.0
    radius = 1.0
    bound = math.asin(radius / center_norm)

    tangent = [
        (center_norm**2 - radius**2) / center_norm,
        radius * math.sqrt(center_norm**2 - radius**2) / center_norm,
    ]
    center_2d = [center_norm, 0.0]
    assert math.isclose(norm([tangent[0] - center_norm, tangent[1]]), radius)
    assert math.isclose(angle(center_2d, tangent), bound, rel_tol=1e-12)

    rng = random.Random(68)
    for dimension in (2, 3, 16):
        center = [center_norm] + [0.0] * (dimension - 1)
        for _ in range(1_000):
            direction = [rng.gauss(0.0, 1.0) for _ in range(dimension)]
            direction_norm = norm(direction)
            sample_radius = radius * rng.random() ** (1.0 / dimension)
            point = [
                center[index] + sample_radius * direction[index] / direction_norm
                for index in range(dimension)
            ]
            assert angle(center, point) <= bound + 1e-12




def check_disagreement_accompanies_double_flip(samples: int = 50_000) -> None:
    """2-토큰 가족 무작위 탐색 — '이중 반전(g10·g01 모두 g_pi와 역방향) ∧ 셀 일치
    (cos(g10,g01)>0.9)' 사례가 없음을 수치 확인 (T2 방어의 재현 스크립트).

    함의: 최소 가족에서 두 one-sided가 동시에 틀릴 때는 서로 불일치한다 —
    셀 불일치는 unreliability 신호가 된다 (수치 관찰; 일반 구조에 대한 증명 아님).
    """
    import random as _random

    rng = _random.Random(7)

    def gradients(q, s0, s1, qb, sb0, sb1):
        def z(a1, a2):
            s = s1 if a1 == 1 else s0
            return (a1 - q, a2 - s)

        out = {k: [0.0, 0.0] for k in ("pi", "00", "10", "01", "11")}
        for a1 in (0, 1):
            p_pi = q if a1 == 1 else 1 - q
            p_b = qb if a1 == 1 else 1 - qb
            s_pi = s1 if a1 == 1 else s0
            s_b = sb1 if a1 == 1 else sb0
            for a2 in (0, 1):
                reward = a2
                z1, z2 = z(a1, a2)
                pp = p_pi * (s_pi if a2 == 1 else 1 - s_pi)
                pb = p_b * (s_b if a2 == 1 else 1 - s_b)
                out["pi"][0] += pp * reward * z1
                out["pi"][1] += pp * reward * z2
                r1 = p_pi / p_b
                r2 = (s_pi if a2 == 1 else 1 - s_pi) / (s_b if a2 == 1 else 1 - s_b)
                out["00"][0] += pb * r1 * reward * z1
                out["00"][1] += pb * r2 * reward * z2
                out["10"][0] += pb * r1 * reward * z1
                out["10"][1] += pb * r1 * r2 * reward * z2
                out["01"][0] += pb * r1 * r2 * reward * z1
                out["01"][1] += pb * r2 * reward * z2
                out["11"][0] += pb * r1 * r2 * reward * z1
                out["11"][1] += pb * r1 * r2 * reward * z2
        return out

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def cosine(a, b):
        na, nb = norm(a), norm(b)
        return dot(a, b) / (na * nb) if na > 0 and nb > 0 else 0.0

    violations = 0
    for _ in range(samples):
        eps = rng.choice([0.05, 0.1, 0.2])
        q = min(max(rng.uniform(0.2, 0.8), 1e-3), 1 - 1e-3)
        qb = min(max(q + rng.uniform(-eps, eps), 1e-3), 1 - 1e-3)
        s0 = rng.uniform(0.2, 0.8)
        s1 = rng.uniform(0.2, 0.8)
        sb0 = min(max(s0 + rng.uniform(-eps, eps), 1e-3), 1 - 1e-3)
        sb1 = min(max(s1 + rng.uniform(-eps, eps), 1e-3), 1 - 1e-3)
        g = gradients(q, s0, s1, qb, sb0, sb1)
        assert abs(g["11"][0] - g["pi"][0]) < 1e-9  # g11 ≡ g_pi 항등
        assert abs(g["11"][1] - g["pi"][1]) < 1e-9
        if norm(g["pi"]) < 1e-6:
            continue
        both_flip = (cosine(g["10"], g["pi"]) < -0.3
                     and cosine(g["01"], g["pi"]) < -0.3)
        agree = cosine(g["10"], g["01"]) > 0.9
        if both_flip and agree:
            violations += 1
    assert violations == 0, f"이중 반전+일치 사례 {violations}건 — T2 방어 재검토 필요"


def report_cells() -> None:
    print("ε     example        KL      gπ       g00      g10      g01      g11     KL/ε²")
    for epsilon in (0.4, 0.1, 0.01, 0.001):
        for example in ("prefix_only", "future_only"):
            cells = is_cells(example, epsilon)
            kl = trajectory_kl(example, epsilon)
            print(
                f"{epsilon:<5} {example:<14} {kl:8.5f} "
                f"{cells['pi']:+.5f} {cells['00']:+.5f} {cells['10']:+.5f} "
                f"{cells['01']:+.5f} {cells['11']:+.5f} {kl / epsilon**2:7.3f}"
            )


def main() -> None:
    check_sign_reversals()
    check_ranking_reversal()
    check_leave_one_out_unbiasedness()
    check_angular_radius()
    check_disagreement_accompanies_double_flip()
    report_cells()
    print(
        "PASS: sign reversals, IS cells, relative error -1, ranking flip, "
        "K=2/4/8 normalization, LOO unbiasedness, "
        "confidence-ball angular radius, and double-flip⇒disagreement (50k)"
    )


if __name__ == "__main__":
    main()


# ---- 일반 K 그룹 정규화 검증 (T3, PAPER_REVIEW §3) ----------------------
# 성공률 1/2 대칭 구성에서 그룹 표준화 계수 m_K = E[A_1 | r_1=1]을 이항 합으로
# 정확 계산한다. 혼합 그룹에서 성공 표본의 표준화 advantage는 결정론적으로
# 양수이므로 m_K > 0 (K>=2) — 부호 반전이 모든 K에서 보존된다는 보조정리의
# 수치 대응물. LOO는 m_K = 1/2로 상수.
def _groupnorm_mk(K: int, p: float = 0.5) -> float:
    from math import comb, sqrt
    mk = 0.0
    for j in range(K):                      # 나머지 K-1개 중 성공 j개
        m = j + 1                           # 그룹 성공 수 (r_1=1 포함)
        if m == K:                          # 전원 성공 — 표준편차 0, 기여 0
            continue
        q = comb(K - 1, j) * p**j * (1 - p)**(K - 1 - j)
        mu = m / K
        mk += q * (1 - mu) / sqrt(mu * (1 - mu))
    return mk


def check_groupnorm_general_k() -> None:
    ks = list(range(2, 65)) + [128, 256, 512, 1024]
    vals = {K: _groupnorm_mk(K) for K in ks}
    bad = [K for K, v in vals.items() if not v > 0]
    assert not bad, f"m_K <= 0 발생: {bad}"
    print(f"[groupnorm-general-K] m_K > 0 확인: K=2..64,128,256,512,1024 "
          f"(예: m_2={vals[2]:.4f}, m_8={vals[8]:.4f}, m_1024={vals[1024]:.4f})")


if __name__ == "__main__":
    check_groupnorm_general_k()
