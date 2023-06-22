import numpy as np
from dtaidistance import dtw
from ..ai_models import *
from ..ai_explainers import global_explanations_data


class AICommenter(object):
    """Based on AI predictions, AICommenter generate high-level AI comments that are more understandable and useful for users in practice."""
    def __init__(self, conference, writingInput, predict_outputs, inputTexts):
        self.conference = conference
        self.predict_outputs = predict_outputs
        self.writingInput = writingInput
        self.inputTexts = inputTexts
        self.review_summary = {}
        self.aspect_model = TfidfAspectModel(self.conference)
        self.revision_comment_template = {
            "shorter_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=shorter-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>short than 20%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. The average length is {ave_word} words.</p><br>",
            "longer_length": "&nbsp;&nbsp;<p class='comments' id={id} class-id=longer-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The sentence is <strong>longer than 80%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. The average length is {ave_word} words.</p><br>",
            "label_change": "&nbsp;&nbsp;<p class='comments' id={id} class-id=label-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: Based on the aspect <strong>labels' percentage and order</strong> of your abstract, it is suggested to write your <strong>{aspect_new}</strong> at this sentence, rather than describing <strong>{aspect_origin}</strong> here.</p><br>",
            "low_score": "&nbsp;&nbsp;<p class='comments' id={id} class-id=score-{id}><strong>-</strong> <span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'><strong>{sentence}</strong></span>: The style quality score of {sentence} is <strong>lower than 20%</strong> of the published <strong>'{label}'</strong>-labeled sentences in {conference} conference. Indicating the writing style might not match well with this conference.<br>"
        }
        self.global_explanations_data = global_explanations_data
            
    def _ai_comment_score_classifier(self, raw_score, score_benchmark):
        """The lower scores get higher labels."""
        if raw_score >= score_benchmark[4]:
            score_label = 1
        if raw_score >= score_benchmark[3] and raw_score < score_benchmark[4]:
            score_label = 2
        if raw_score >= score_benchmark[1] and raw_score < score_benchmark[3]:
            score_label = 3
        if raw_score >= score_benchmark[0] and raw_score < score_benchmark[1]:
            score_label = 4
        if raw_score < score_benchmark[0]:
            score_label = 5
        return score_label

    def _ai_comment_analyzer(self):

        ###### Benchmark Scores ######
        abstract_score_benchmark = self.global_explanations_data[self.conference]['abstract_score_range']
        sentence_score_benchmark = self.global_explanations_data[self.conference]['sentence_score_range']
        sentence_length_benchmark = self.global_explanations_data[self.conference]['sentence_length']

        ###### Quality Scores ######
        sentence_raw_score = self.predict_outputs["outputs_perplexity_list"]
        abstract_score = self._ai_comment_score_classifier(np.mean(sentence_raw_score), abstract_score_benchmark)

        ###### Aspects Labels & Patterns ######
        aspect_list = self.predict_outputs["outputs_predict_list"]  # aspect_feedback
        aspect_distribution_benchmark = self.aspect_model.predict(aspect_list)
        aspect_distribution_benchmark = aspect_distribution_benchmark.aspect_sequence

        length_revision = ""
        structure_revision = ""
        style_revision = ""

        # Aspects Labels Analysis ######: 
        # DTW algorithm: https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-distance-measure-between-two-time-series;
        # https://towardsdatascience.com/an-illustrative-introduction-to-dynamic-time-warping-36aa98513b98
        distance, paths = dtw.warping_paths(
            aspect_list, aspect_distribution_benchmark)
        best_path = dtw.best_path(paths)
        change_label_dict = {}
        for path in best_path:
            if aspect_list[path[0]] != aspect_distribution_benchmark[path[1]]:
                change_label_dict[path[0]] = path[1]

        quality_score = 0
        ###### Length and Score Analysis ######
        for n in range(len(sentence_raw_score)):
            sentence_score = self._ai_comment_score_classifier(
                sentence_raw_score[n], sentence_score_benchmark[diversity_model_label_mapping[aspect_list[n]]])
            sentence_length = self._ai_comment_score_classifier(len(counting_tokens(
                self.inputTexts[n])), sentence_length_benchmark[diversity_model_label_mapping[aspect_list[n]]])

            if n in change_label_dict.keys():
                k = change_label_dict[n]
                structure_revision += self.revision_comment_template["label_change"].format(
                    id=f"{n}", sentence=f"S{n+1}", aspect_origin=diversity_model_label_mapping[aspect_list[n]], aspect_new=diversity_model_label_mapping[aspect_distribution_benchmark[k]])
                self.review_summary.setdefault(n, []).append(
                    f"aspect-{diversity_model_label_mapping[aspect_list[n]]}-{diversity_model_label_mapping[aspect_distribution_benchmark[k]]}")

            if sentence_score < 2:
                style_revision += self.revision_comment_template["low_score"].format(
                    id=f"{n}", sentence=f"S{n+1}", label=diversity_model_label_mapping[aspect_list[n]], conference=self.conference)
                self.review_summary.setdefault(n, []).append(
                    f"quality-{sentence_raw_score[n]}")

            if sentence_length > 4:
                length_revision += self.revision_comment_template["shorter_length"].format(
                    id=f"{n}", sentence=f"S{n+1}", conference=self.conference, label=diversity_model_label_mapping[aspect_list[n]], ave_word=self.global_explanations_data[self.conference]["sentence_length"][diversity_model_label_mapping[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(
                    f"short-{len(counting_tokens(self.inputTexts[n]))}")

            if sentence_length < 2:
                length_revision += self.revision_comment_template["longer_length"].format(
                    id=f"{n}", sentence=f"S{n+1}", conference=self.conference, label=diversity_model_label_mapping[aspect_list[n]], ave_word=self.global_explanations_data[self.conference]["sentence_length"][diversity_model_label_mapping[aspect_list[n]]][2])
                self.review_summary.setdefault(n, []).append(
                    f"long-{len(counting_tokens(self.inputTexts[n]))}")

            quality_score += sentence_score
        abstract_quality_score = quality_score / len(sentence_raw_score)

        self.review_summary["abstract_score_benchmark"] = self.global_explanations_data[self.conference]['abstract_score_range']
        self.review_summary["sentence_score_benchmark"] = self.global_explanations_data[
            self.conference]['sentence_score_range'][diversity_model_label_mapping[aspect_list[n]]]
        self.review_summary["sentence_length_benchmark"] = self.global_explanations_data[
            self.conference]['sentence_length'][diversity_model_label_mapping[aspect_list[n]]]
        self.review_summary["prediction_label"] = diversity_model_label_mapping[aspect_list[n]]
        aspect_keys = "".join(list(map(str, aspect_distribution_benchmark)))
        self.review_summary["aspect_distribution_benchmark"] = self.global_explanations_data[self.conference]["Aspect_Patterns_dict"][aspect_keys]

        if len(length_revision) == 0:
            length_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your sentence lengths look good to me. Great job!</p>"

        if len(structure_revision) == 0:
            structure_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your abstract structures look good to me. Great job!</p>"

        if len(style_revision) == 0:
            style_revision = "&nbsp;&nbsp;<p><strong>-</strong>Your writing styles look good to me. Great job!</p>"

        feedback_improvement = f"<br> <p style='color:#1B5AA2;font-weight:bold'> Structure Suggestions:</p> {structure_revision}" + \
                               f"<br> <p style='color:#1B5AA2;font-weight:bold'> Style Suggestions:</p> {style_revision} {length_revision}"

        abstract_structure_score = 5 - 0.5 * len(change_label_dict.keys())
        overall_score = (abstract_structure_score + abstract_quality_score) / 2
        print(f"===>>> The overall_score = {overall_score}, with abstract_structure_score = {abstract_structure_score} and abstract_quality_score = {abstract_quality_score}.")
        analysis_outputs = {
            "abstract_score": overall_score,
            "abstract_structure_score": abstract_structure_score,
            "abstract_quality_score": abstract_quality_score,
            "instance_results": feedback_improvement
        }

        return analysis_outputs

    def ai_comment_generation(self):
        """Generate the AI comments, which include:
        1) the a brief statistic of the selected conference;
        2) the generated writing score;
        3) how to improvement for specific sentences.
        """
        analysis_outputs = self._ai_comment_analyzer()
        comment_conference_intro = f"Nice! I'm comparing your abstract with <strong>{self.global_explanations_data[self.conference]['paper_count']} published {self.conference}</strong> abstracts."
        comment_overall_score = f"<br><br><p class='overall-score'> Your <strong>Overall Score</strong>=<strong>{analysis_outputs['abstract_score']:0.2f}</strong> (out of 5) by averaging: <br> Structure Score = {analysis_outputs['abstract_structure_score']:0.2f} and Style Score = {analysis_outputs['abstract_quality_score']:0.2f}. </p> "
        comment_improvements = "<br>" + analysis_outputs['instance_results'] if len(analysis_outputs['instance_results']) > 0 else "Your current writing looks good to me. Great job!"
        return comment_conference_intro + comment_overall_score + comment_improvements


    def _explaining_ai_comment_template(self, idx, review, convxai_global_status_track):

        review_type = review.split("-")[0]

        if review_type == "long":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <span class='text-danger font-weight-bold'>shorten sentence length</span>:
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern1: Similar Examples (rank: short).</span> Refer to similar short examples for rewriting.  
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern2: Rewrite while keeping important_words.</span> Find Important Words, then keep them during rewriting to keep the correct aspects.
            <br><br><p class='text-danger font-weight-bold'>Useful XAIs are:</p>
            """

        elif review_type == "short":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <span class='text-danger font-weight-bold'>lengthen sentence length</span>:
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern1: Similar Examples (rank: long).</span> Refer to similar long examples for rewriting.  
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern2: Rewrite while keeping important_words.</span> Find Important Words, then keep them during rewriting to keep the correct aspects.
            <br><br><p class='text-danger font-weight-bold'>Useful XAIs are:</p>
            """

        elif review_type == "quality":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <span class='text-danger font-weight-bold'>improve sentence quality</span>:
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern1: Counterfactual Explanation (use same label).</span> Ask GPT-3 model to paraphrase the original sentence.  
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern2: Similar Examples (rank: quality_score).</span> Refer to similar examples with high quality scores.
            <br><br><p class='text-danger font-weight-bold'>Useful XAIs are:</p>
            """

        elif review_type == "aspect":
            response = f"""<span style='background-color: #6D589B; font-weight: bold; border-radius: 3px; color:white'>S{idx+1}</span>: XAI use patterns to <span class='text-danger font-weight-bold'>rewrite into target-label</span>:
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern1: Counterfactual Explanation (use target-label).</span> Ask GPT-3 model to rewrite the sentence into the target aspect.  
            <br><br><span style='color:#1B5AA2;font-weight:bold'>Pattern2: Similar Examples (label: target-label, rank: quality_score).</span> Refer to similar examples with the target labels and high quality scores.
            <br><br><p class='text-danger font-weight-bold'>Useful XAIs are:</p>
            """

        return [review_type, response]

    def explaining_ai_comment_instance(self, writingIndex, convxai_global_status_track):
        response = []
        for idx in writingIndex:
            if int(idx) in self.review_summary.keys():
                for item in self.review_summary[int(idx)]:
                    response.extend(
                        self._explaining_ai_comment_template(int(idx), item, convxai_global_status_track))
        if len(response) == 0:
            response = [
                "none", "Your writing looks good to us! <br><br><strong>To improve</strong>, you can ask for explanation questions below:<br><br>"]
        return response
