import argparse
import json
import subprocess


def get_attribute_filter_config(attribute_kind: str, sub_sample_seed=1):
    def escape_quotes(s: str) -> str:
        return s.replace('"', '\\"')

    sub_sample_filter = (
        "weka.filters.supervised.instance.SpreadSubsample"
        + " -M 1.0 -X 0.0 -S {}".format(sub_sample_seed)
    )

    if attribute_kind == "only-ir":
        filter_remove = "weka.filters.unsupervised.attribute.Remove -V -R 7,23,24,26"
        return 'weka.filters.MultiFilter -F "{}" -F "{}"'.format(
            filter_remove, sub_sample_filter
        )

    elif attribute_kind == "only-struct":
        filter_remove = "weka.filters.unsupervised.attribute.Remove -R 1-3,22-25"
        return 'weka.filters.MultiFilter -F "{}" -F "{}"'.format(
            filter_remove, sub_sample_filter
        )

    elif attribute_kind == "all":
        filter_remove = "weka.filters.unsupervised.attribute.Remove -R 1-3,22,25"
        return 'weka.filters.MultiFilter -F "{}" -F "{}"'.format(
            filter_remove, sub_sample_filter
        )

    elif attribute_kind == "auto":
        filter_remove = "weka.filters.unsupervised.attribute.Remove -R 1-3,22,25"

        # parameters for best attribute selection
        sel_first = "weka.attributeSelection.CfsSubsetEval -P 1 -E 1"
        sel_second = "weka.attributeSelection.BestFirst -D 1 -N 5"

        select_filter = 'weka.filters.supervised.attribute.AttributeSelection -E "{}" -S "{}"'.format(
            sel_first, sel_second
        )

        return 'weka.filters.MultiFilter -F "{}" -F "{}" -F "{}"'.format(
            filter_remove, sub_sample_filter, escape_quotes(select_filter)
        )


CLASSIFIER = {
    "zeroR": "weka.classifiers.rules.ZeroR",
    "naiveBayes": "weka.classifiers.bayes.NaiveBayes",
    "j48": "weka.classifiers.trees.J48",
    "randomForest": "weka.classifiers.trees.RandomForest",
}


OUTPUT_PREDICTION = {
    "csv": "weka.classifiers.evaluation.output.prediction.CSV",
    "txt": "weka.classifiers.evaluation.output.prediction.PlainText",
}


def get_classifier_train_options(classifier_name: str) -> list:
    if classifier_name == "j48":
        s = CLASSIFIER[classifier_name] + " -- -C 0.25 -M 2"
        return s.split(" ")

    elif classifier_name == "randomForest":
        s = (
            CLASSIFIER[classifier_name]
            + " -- -P 100 -I 100"
            + " -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1"
        )
        return s.split(" ")

    else:
        return [CLASSIFIER[classifier_name]]


DEFAULT_SETTINGS = {
    "weka_jar": "/Users/micha/Programs/weka-3-8-1/weka.jar",
    "sub_sample_seed": 1,
    "classifier": "zeroR",
    "attribute_config": "all",
    "execution_mode": "train",
    "train_file": None,
    "model_file": None,
    "prediction_output_file": None,
    "prediction_output_format": "csv",
    "test_file": None,
}


def build_argument_parser():
    parser = argparse.ArgumentParser(prog="run_weka")

    parser.add_argument(
        "-c",
        "--classifier",
        type=str,
        choices=CLASSIFIER.keys(),
        help="Classifier to use",
    )
    parser.add_argument(
        "--attribute-config",
        type=str,
        choices=["only-ir", "only-struct", "all", "auto"],
        help="Attribute configuration",
    )
    parser.add_argument(
        "-e",
        "--execution-mode",
        type=str,
        choices=["train", "test", "predict"],
        help="",
    )

    # important files
    parser.add_argument("--train-file", type=str, help="Input training file")
    parser.add_argument(
        "--model-file", type=str, help="Path to store/load trained model"
    )
    parser.add_argument("--test-file", type=str, help="Input testing/prediction file")

    # settings handling
    parser.add_argument(
        "--settings-input-file", type=str, help="Read settings from file"
    )
    parser.add_argument(
        "--settings-output-file", type=str, help="Save settings in file"
    )

    # weka execution
    parser.add_argument(
        "--run", default=False, action="store_true", help="Actually **RUN** weka"
    )

    # prediction
    parser.add_argument(
        "--prediction-output-format",
        type=str,
        choices=OUTPUT_PREDICTION.keys(),
        help="Prediction file format",
    )
    parser.add_argument(
        "--prediction-output-file", type=str, help="File to store prediction output"
    )

    # misc
    parser.add_argument("--sub-sample-seed", type=int, help="Seed for sub sampling")
    parser.add_argument("--weka-jar", type=str, help="path to weka jar")

    return parser


def get_program_arguments(settings: dict) -> list:
    if settings["execution_mode"] == "train":
        filter_config = get_attribute_filter_config(settings["attribute_config"])
        classifier_config = get_classifier_train_options(settings["classifier"])

        args = [
            "java",
            "-cp",
            settings["weka_jar"],
            "weka.classifiers.meta.FilteredClassifier",
            "-no-cv",
            "-d",
            settings["model_file"],
            "-t",
            settings["train_file"],
            "-F",
            filter_config,
            "-W",
        ] + classifier_config

        return args

    elif settings["execution_mode"] == "test":
        args = [
            "java",
            "-cp",
            settings["weka_jar"],
            CLASSIFIER[settings["classifier"]],
            "-l",
            settings["model_file"],
            "-T",
            settings["test_file"],
        ]

        return args

    elif settings["execution_mode"] == "predict":
        # Note: '-suppress' suppresses the output of prediction results
        #       on commandline
        classification_config = "{} -p 2,3 -file {} -suppress".format(
            OUTPUT_PREDICTION[settings["prediction_output_format"]],
            settings["prediction_output_file"],
        )

        args = [
            "java",
            "-cp",
            settings["weka_jar"],
            # breaks on windows: CLASSIFIER[settings["classifier"]],
            "weka.classifiers.meta.FilteredClassifier",
            "-l",
            settings["model_file"],
            "-T",
            settings["test_file"],
            "-classifications",
            classification_config,
        ]

        return args

    return []


def apply_commandline_options(input_settings: dict, opts) -> dict:
    settings = dict(input_settings)

    for k, v in vars(opts).items():
        if k in settings:
            if v:
                # only overwrite, if value not NONE
                settings[k] = v

    return settings


def main_command_line():
    parser = build_argument_parser()
    options = parser.parse_args()

    settings = dict(DEFAULT_SETTINGS)

    if options.settings_input_file:
        with open(options.settings_input_file, "r") as fp:
            read_settings = json.load(fp)
            # merge settings
            settings.update(read_settings)

    # apply commandline options
    settings = apply_commandline_options(settings, options)

    if options.settings_output_file:
        with open(options.settings_output_file, "w") as fp:
            json.dump(settings, fp, indent=2, sort_keys=True)

    if not options.run:
        # show the settings in json format, so it can be piped to file
        print(json.dumps(settings, indent=2, sort_keys=True))

    if options.run:
        program_arguments = get_program_arguments(settings)
        print("weka commandline: {} ".format(program_arguments))
        subprocess.run(program_arguments)


###############################################################################


if __name__ == "__main__":
    main_command_line()
