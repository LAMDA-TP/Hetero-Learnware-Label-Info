import argparse
from .framework import *

def run_workflow(task_type):
    """Runs the workflow for the given task type and returns results."""
    workflow = Workflow(task_type=task_type, regenerate_flag=False)
    dataset_list = DATASET_LIST[task_type]

    results_dict = {}
    results_dict_varied_labeled_data = {}
    for task_name in dataset_list:
        print(f"Processing task: {task_name}")
        workflow.load_market(task_name)
        results_dict[task_name] = workflow.run_our_method()
        results_dict_varied_labeled_data[task_name] = workflow.run_our_method_varied_labeled_data()
    
    workflow.aggregate_results(results_dict)
    workflow.plot_results(results_dict_varied_labeled_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, choices=["classification", "regression"], required=True,
                        help="Task type: 'classification' or 'regression'.")
    args = parser.parse_args()
    
    run_workflow(args.task_type)

if __name__ == "__main__":
    main()