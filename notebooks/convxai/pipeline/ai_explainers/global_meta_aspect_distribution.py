
def explain_aspect_distribution(conference, global_explanations_data):
    """Global Explanations
    """
    explanation1 = """We use the Research Aspects Model to generate <strong>aspect sequences</strong> of all 9935 paper abstracts. Then we cluster these sequences into <strong>five patterns</strong> as below. We compare your writing with these patterns for review.
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

    explanation2 = f""" 
    <table class="demo">
        <caption><br></caption>
        <thead>
        <tr>
            <th>Types</th>
            <th>Patterns</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>Pattern1</td>
            <td>{list(global_explanations_data[conference]['Aspect_Patterns_dict'].values())[0]}</td>
        </tr>
        <tr>
            <td>Pattern2&nbsp;</td>
            <td>{list(global_explanations_data[conference]['Aspect_Patterns_dict'].values())[1]}</td>
        </tr>
        <tr>
            <td>Pattern3&nbsp;</td>
            <td>{list(global_explanations_data[conference]['Aspect_Patterns_dict'].values())[2]}</td>
        </tr>
        <tr>
            <td>Pattern4</td>
            <td>{list(global_explanations_data[conference]['Aspect_Patterns_dict'].values())[3]}</td>
        </tr>
        <tr>
            <td>Pattern5</td>
            <td>{list(global_explanations_data[conference]['Aspect_Patterns_dict'].values())[4]}</td>
        </tr>
        <tbody>
    </table>
    """
    
    return explanation1 + explanation2

