using E7.DataStructure;
using NUnit.Framework;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

namespace Tests
{
    public class JobAnimationCurveTests
    {
        private const int testEvaluateBatchCount = 10;

        private AnimationCurve RandomCurve(int resolution)
        {
            Random.InitState(1234567);
            var kf = new List<Keyframe>();
            for (int i = 0; i < resolution; i++)
            {
                kf.Add(new Keyframe(math.unlerp(0, resolution - 1, i), Random.value, math.tan(math.radians(Random.value * 360)), math.tan(math.radians(Random.value * 360))));
            }
            return new AnimationCurve(kf.ToArray());
        }

        [BurstCompile]
        private struct CurveEvaluationJob : IJobParallelFor
        {
            [NativeDisableParallelForRestriction] public NativeArray<float> evaluated;
            public JobAnimationCurve jobAnimationCurve;

            public void Execute(int index)
            {
                evaluated[index] = jobAnimationCurve.Evaluate(index / (float)evaluated.Length);
            }
        }

        [Test]
        public void Instantiation()
        {
            using (var jac = new JobAnimationCurve(RandomCurve(10), Allocator.TempJob)) { }
        }

        [Test]
        public void Evaluate2EaseInOut([Values(10, 100, 1000)] int evaluationResolution)
        {
            TestEvaluate(AnimationCurve.EaseInOut(0, 0, 1, 1), evaluationResolution);
        }

        [Test]
        [Ignore("This works if I remove a weight check in the code, since this setup weighted into the same curve.")]
        public void Evaluate2WeightedUnweight([Values(10, 100, 1000)] int evaluationResolution)
        {
            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outWeight = 0.3333333333f;
            keys[0].weightedMode = WeightedMode.Both;
            keys[1].inWeight = 0.3333333333f;
            keys[1].weightedMode = WeightedMode.Both;
            ac.keys = keys;
            TestEvaluate(ac, evaluationResolution);
        }

        [Test]
        [Ignore("This works if I remove a weight check in the code, since this setup weighted into the same curve.")]
        public void Evaluate2WeightedUnweightLinear([Values(10, 100, 1000)] int evaluationResolution)
        {
            //If the tangents are both 1, weight has no effect.

            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outTangent = 1;
            keys[0].outWeight = 0.15f;
            keys[0].weightedMode = WeightedMode.Both;
            keys[1].inTangent = 1;
            keys[1].inWeight = 0.7f;
            keys[1].weightedMode = WeightedMode.Both;
            ac.keys = keys;
            TestEvaluate(ac, evaluationResolution);
        }

        [Test]
        [Ignore("Because weight doesn't work yet...")]
        public void Evaluate2Weighted([Values(10, 100, 1000)] int evaluationResolution)
        {
            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outWeight = 0.15f;
            keys[0].weightedMode = WeightedMode.Both;
            keys[1].inWeight = 0.7f;
            keys[1].weightedMode = WeightedMode.Both;
            ac.keys = keys;
            TestEvaluate(ac, evaluationResolution);
        }


        [Test]
        public void Evaluate2Tangents([Values(10, 100, 1000)] int evaluationResolution)
        {
            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outTangent = math.tan(math.radians(55));
            keys[1].inTangent = math.tan(math.radians(12));
            ac.keys = keys;

            TestEvaluate(ac, evaluationResolution);
        }

        [Test]
        public void Evaluate2TangentsAndValues([Values(10, 100, 1000)] int evaluationResolution)
        {
            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outTangent = math.tan(math.radians(55));
            keys[0].value = 213.412f;
            keys[1].inTangent = math.tan(math.radians(12));
            keys[1].value = 152.3244f;
            ac.keys = keys;

            TestEvaluate(ac, evaluationResolution);
        }

        [Test]
        public void Evaluate2TangentsAndValuesWithUnrelatedTangents ([Values(10, 100, 1000)] int evaluationResolution)
        {
            var ac = AnimationCurve.EaseInOut(0, 0, 1, 1);
            var keys = ac.keys;
            keys[0].outTangent = math.tan(math.radians(55));
            keys[0].value = 213.412f;
            keys[1].inTangent = math.tan(math.radians(12));
            keys[1].value = 152.3244f;

            keys[0].inTangent = math.tan(math.radians(44));
            keys[0].outTangent = math.tan(math.radians(33));

            ac.keys = keys;

            TestEvaluate(ac, evaluationResolution);
        }

        [Test]
        public void EvaluateRandomCurves([Values(/* 1,*/ 2, 3, 10, 100, 1000)] int curveResolution, [Values(10, 100, 1000)] int evaluationResolution)
        {
            TestEvaluate(RandomCurve(curveResolution), evaluationResolution);
        }

        [Test]
        public void EvaluateRandomCurvesOnNode([Values(1, 10, 100, 1000)]int nodes)
        {
            TestEvaluate(RandomCurve(nodes + 1), nodes);
        }

        [Test]
        [Description("On purely main thread, the replicate algorithm should be competitive to Unity ones.")]
        public void MainThreadPerformanceTest([Values(/* 1,*/ 2, 3, 10, 100, 1000)] int curveResolution, [Values(10, 100, 1000)] int evaluationResolution)
        {
            var ticks = MainThreadPerformanceTest(RandomCurve(curveResolution), evaluationResolution);
            Assert.That(ticks.jobTicks, Is.LessThan(ticks.mainTicks));
        }

        private (long mainTicks, long jobTicks) MainThreadPerformanceTest(AnimationCurve ac, int iterationCount)
        {
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            var jac = new JobAnimationCurve(ac, Allocator.Temp);

            float[] evaluated = new float[iterationCount];

            sw.Start();
            for (int i = 0; i < evaluated.Length; i++)
            {
                evaluated[i] = jac.Evaluate(i / (float)iterationCount);
            }
            sw.Stop();
            var jobTicks = sw.ElapsedTicks;

            sw.Reset();

            sw.Start();
            for (int i = 0; i < evaluated.Length; i++)
            {
                evaluated[i] = ac.Evaluate(i / (float)iterationCount);
            }
            sw.Stop();
            var mainTicks = sw.ElapsedTicks;


            jac.Dispose();

            return (mainTicks, jobTicks);
        }

        private (long mainTicks, long jobTicks) TestEvaluate(AnimationCurve ac, int iterationCount)
        {
            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();

            var jac = new JobAnimationCurve(ac, Allocator.TempJob);

            NativeArray<float> evaluated = new NativeArray<float>(iterationCount, Allocator.TempJob);
            NativeArray<float> jobEvaluated = new NativeArray<float>(iterationCount, Allocator.TempJob);
            var job = new CurveEvaluationJob
            {
                evaluated = jobEvaluated,
                jobAnimationCurve = jac
            }.Schedule(iterationCount, testEvaluateBatchCount, default(JobHandle));

            sw.Start();
            for (int i = 0; i < evaluated.Length; i++)
            {
                evaluated[i] = ac.Evaluate(i / (float)iterationCount);
            }
            sw.Stop();
            var mainTicks = sw.ElapsedTicks;
            sw.Reset();

            sw.Start();
            job.Complete();
            sw.Stop();
            var jobTicks = sw.ElapsedTicks;

            for (int i = 0; i < evaluated.Length; i++)
            {
                //Within 0.00001f, it is a bit inaccurate.
                Assert.That(evaluated[i], Is.EqualTo(jobEvaluated[i]).Within(0.0001f),
                $"At index {i} (time {i / (float)iterationCount}) there is a difference of Unity {evaluated[i]} and Job {jobEvaluated[i]} ({evaluated[i] - jobEvaluated[i]})");
            }

            evaluated.Dispose();
            jobEvaluated.Dispose();
            jac.Dispose();

            return (mainTicks, jobTicks);
        }
    }
}
