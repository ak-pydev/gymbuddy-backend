# Gym Domain Framing for Paper

## Near-OOD Deployment Domain Definition

The gym_2d dataset represents a near-OOD (Out-of-Distribution) deployment scenario that is conceptually critical to our safety-focused evaluation framework. Unlike true OOD data where action semantics change entirely, gym_2d shares the same fundamental task as NTU-120—skeleton-based action recognition—but operates under degraded signal conditions characteristic of real-world fitness applications.

### Key Characteristics

| Property | NTU-120 (ID) | Gym Domain (Near-OOD) |
|----------|--------------|----------------------|
| Task | Action recognition | Same |
| Skeleton quality | Lab-grade depth sensors | Commercial webcam/phone |
| Occlusions | Controlled | Frequent (equipment, body parts) |
| Viewpoints | Fixed protocols | Arbitrary user angles |
| Noise | Minimal | Pose estimation artifacts |

### Why This Matters

The gym domain exemplifies the *same task, worse signal* paradigm:

1. **Pose estimation noise**: 2D skeleton extraction from RGB video introduces coordinate jitter, missing joints, and temporal discontinuities absent in depth-sensor data.

2. **Occlusion patterns**: Gym equipment (barbells, benches, machines) creates systematic occlusions during exercise movements—precisely when accurate recognition matters most.

3. **Distribution shift**: While action labels overlap (squats, lunges, deadlifts), the feature distribution shifts due to sensor modality, environmental conditions, and user variability.

### Safety Implications

This near-OOD framing justifies our uncertainty-centric evaluation:

> **"When the signal degrades, the model should know it doesn't know."**

A model deployed in gym settings must:
- Recognize when input quality undermines reliable prediction
- Abstain from giving feedback when confidence is unwarranted  
- Maintain safety by avoiding harmful corrections based on uncertain predictions

This is why **uncertainty matters more than raw accuracy** in the gym domain—a confidently wrong correction could lead to injury, while appropriate abstention maintains trust and safety.

---

## Paragraph for Paper (Camera-Ready Version)

> We evaluate transfer to a gym-based deployment domain that represents a realistic near-OOD scenario. Unlike true out-of-distribution settings where action semantics differ entirely, the gym domain shares the same recognition task as our training data but operates under significantly degraded signal conditions. Skeleton sequences extracted from consumer-grade RGB video exhibit pose estimation noise, frequent occlusions from equipment and body parts, and viewpoint variability absent in laboratory datasets. This "same task, worse signal" paradigm is precisely where uncertainty quantification becomes critical: rather than pursuing marginal accuracy gains, we demonstrate that epistemic uncertainty enables the model to recognize when input quality undermines reliable prediction, allowing graceful degradation through principled abstention rather than confident errors that could compromise user safety.
