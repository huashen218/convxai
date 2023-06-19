
def explain_quality_score(conference, global_explanations_data):
    """Global Explanations
    table generator: https://www.rapidtables.com/web/tools/html-table-generator.html
    """
    
    explanation1 = f"""
    We use each sentence's <strong>Perplexity</strong> value (predicted by the GPT-2 model) to compute the <strong>Quality Style Score</strong>. Lower perplexity score means better quality for {conference}.
    <br>
    <br>
    We set the criteria by dividing into five levels based on [20-th, 40-th, 60-th, 80-th] percentiles of all the {conference} papers' perplexity scores.
    <br>
    <br>
    For example, the percentiles = [{global_explanations_data[conference]['abstract_score_range'][0]}, {global_explanations_data[conference]['abstract_score_range'][1]}, {global_explanations_data[conference]['abstract_score_range'][3]}, {global_explanations_data[conference]['abstract_score_range'][4]}]), resulting in the criterion:
    """



    explanation2 = """
    <br>
    <style>
        .demo {
            border:1px solid #EDEDED;
            border-collapse:separate;
            border-spacing:2px;
            padding:5px;
        }
        .demo th {
            border:1px solid #EDEDED;
            padding:5px;
            background:#D6D6D6;
        }
        .demo td {
            border:1px solid #EDEDED;
            text-align:center;
            padding:5px;
            background:#F5F5F5;
        }
    </style>
    """

    explanation3 = f"""
    <table class="demo">
        <caption><br></caption>
        <thead>
        <tr>
            <th>Quality Score</th>
            <th>Perplexity (PPL)</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>1 (lowest)</td>
            <td>{global_explanations_data[conference]['abstract_score_range'][4]} &lt; PPL<br></td>
        </tr>
        <tr>
            <td>2</td>
            <td>{global_explanations_data[conference]['abstract_score_range'][3]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][4]}&nbsp;</td>
        </tr>
        <tr>
            <td>3</td>
            <td>{global_explanations_data[conference]['abstract_score_range'][1]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][3]}&nbsp;</td>
        </tr>
        <tr>
            <td>4</td>
            <td>{global_explanations_data[conference]['abstract_score_range'][0]} &lt; PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][1]}&nbsp;</td>
        </tr>
        <tr>
            <td>5 (highest)</td>
            <td>PPL &lt;= {global_explanations_data[conference]['abstract_score_range'][0]}&nbsp;</td>
        </tr>
        <tbody>
    </table>
    """
    return explanation1 + explanation2 + explanation3

