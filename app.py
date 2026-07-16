from metrics import answer_relevancy
from metrics import faithfulness
from metrics import hallucination
from metrics import bias
from metrics import toxicity
from metrics import geval
from metrics import contextual_precision
from metrics import contextual_recall
from metrics import contextual_relevancy


def main():

    print("=" * 80)
    print("        GenAI QA Automation Framework")
    print("=" * 80)

    metrics = [

        ("Answer Relevancy", answer_relevancy.run),

        ("Faithfulness", faithfulness.run),

        ("Hallucination", hallucination.run),

        ("Bias", bias.run),

        ("Toxicity", toxicity.run),

        ("GEval", geval.run),

        ("Contextual Precision", contextual_precision.run),

        ("Contextual Recall", contextual_recall.run),

        ("Contextual Relevancy", contextual_relevancy.run)

    ]

    passed = 0
    failed = 0

    for metric_name, metric_runner in metrics:

        print("\n")
        print("=" * 80)
        print(f"Running : {metric_name}")
        print("=" * 80)

        try:

            metric_runner()

            print(f"\n✅ {metric_name} Completed Successfully")

            passed += 1

        except Exception as e:

            print(f"\n❌ {metric_name} Failed")
            print(e)

            failed += 1

    print("\n")
    print("=" * 80)
    print("Execution Summary")
    print("=" * 80)

    print(f"Total Metrics : {len(metrics)}")
    print(f"Passed        : {passed}")
    print(f"Failed        : {failed}")

    print("=" * 80)


if __name__ == "__main__":
    main()