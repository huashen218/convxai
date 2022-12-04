/**
 * @file This source code supports the user interface of the ConvXAI system.
 * @copyright Hua Shen 2022
**/

const socket = io.connect('/connection');


/*****************************************/
// Create Log File
/*****************************************/
window.onload = (event) => {
    console.log('The page has fully loaded');
};


/*****************************************/
// Front Page Instruction
/*****************************************/
document.getElementById("start").addEventListener("submit", function (event) {
    event.preventDefault()
    fetch('/init_conversation', {
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify({ text: "start" }),
    }).then(response => response.json()).then(data => {
        console.log(data);
        $("#conversation_id").val(data["conversation_id"]);
        var fragDisplay = document.getElementById("main-content");
        fragDisplay.style.display = "inline";
        var submitDisplay = document.getElementById("display-start");
        submitDisplay.style.display = "none";

    })
});


/*****************************************/
// Select Conferences
/*****************************************/
var target_conference = "";
var target_conference_selection = document.getElementById("target-conference");
target_conference_selection.addEventListener("change", function () {
    target_conference = target_conference_selection.value;
    $('#writing_support_card').animate({
        height: "553pt"
    }, 600);
    $('#editor-form').slideDown(200);
});


/*****************************************/
// Create Editor Panel
/*****************************************/
var Delta = Quill.import('delta');

var quill = new Quill('#editor', {
    theme: 'snow',
    modules: {
        toolbar: [
            [{ 'header': [1, 2, 3, 4, 5, 6, false] }],
            ['bold', 'italic', 'strike', 'underline'],
            ['link', 'blockquote', 'code-block', 'image'],
            [{ list: 'ordered' }, { list: 'bullet' }]
        ]
    },
    toolbar: '#toolbar',
    scrollingContainer: '#editor',
    placeholder: 'Compose an abstract...'
});





function download(content, fileName, contentType) {
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download=fileName;
    a.click();
}


/*****************************************/
// Submit Writing and Update Visualization
/*****************************************/
writing_artifacts = {}
document.getElementById("editor-form").addEventListener("submit", function (event) {
    event.preventDefault();
    $('.btn-submit-loader').css({ "display": "block" });    // Adding waiting logo;
    var writing_model = $("#writing-model").val();      // Identify writing model;
    socket.emit(
        "interact",
        {
            "text": quill.getText(),
            "message_type": "task",
            "writing_model": writing_model,
            "conversation_id": $("#conversation_id").val(),
        }
    );


    const start = Date.now()
    console.log('file', 'writing'+start+'.txt');
    writing_artifacts['text'] = quill.getText()
    var jsonData = JSON.stringify(writing_artifacts);
    download(jsonData, 'convxai_writing_log_'+start+'.txt', 'text/plain');

    return;
});




/*********************************************************/
// Return Both Models' Predictions and Update Visualization
/*********************************************************/
var diversity_text;
var quality_text;

socket.on("task_response", function (data) {
    $('.btn-submit-loader').css({ "display": "none" });     // Removing waiting logo;
    $('.writing-model-descriptions').slideDown("500");
    $('#display-xai').fadeIn(300);

    data = data.text;
    diversity_text = data.slice(0, data.length / 2);
    quality_text = data.slice(data.length / 2, data.length);

    var writing_model = $("#writing-model").val();
    if (writing_model === "" || writing_model === "model-writing-1") {
        $("#writing-model").val("model-writing-1");
        $(`.model-writing-1-instruction`).toggle();
        render_model_output(diversity_text, "model-writing-1");
    } else {
        $(`.model-writing-2-instruction`).toggle();
        render_model_output(quality_text, "model-writing-2");
    }
    initiateConversation();
}
);







/*****************************************/
// Select Conferences
/*****************************************/

$(document).on('click', '[category="predicts"]', function (e) {
    $('[id=' + this.id + ']').toggleClass("predict-selected");
    $('[selection-id=' + this.id + ']').toggle();
    $('[category=predicts]').filter('[id=' + this.id + ']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
    $('.selections').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
});


$(document).on('click', '.comments', function (e) {
    $('[id=' + this.id + ']').toggleClass("predict-selected");
    $('[selection-id=' + this.id + ']').toggle();
    $('[category=predicts]').filter('[id=' + this.id + ']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
    $('.selections').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
});


$(document).on('click', '.selections', function (e) {
    $('[id=' + this.id + ']').toggleClass("predict-selected");
    $('[selection-id=' + this.id + ']').toggle();
    $('[category=predicts]').filter('[id=' + this.id + ']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
    $('.selections').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
});


$(document).on('dblclick', '.comments', function (e) {
    $('[id=' + this.id + ']').toggleClass("predict-selected");
    $('[selection-id=' + this.id + ']').toggle();
    $('[category=predicts]').filter('[id=' + this.id + ']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
    $('.selections').filter('[id=' + this.id + ']').toggleClass('associate-highlight');
    chatBotTextArea = document.querySelector(".chatBot .chatForm #chatTextBox")
    chatBotTextArea.value = "Can you explain this sentence?";
});




/*****************************************/
// Double Click to Select Sentences.
/*****************************************/

function render_model_output(data, writing_model) {
    console.log("render_model_outputppp", data);
    var data_tooltips = ""
    const content_obj = [];
    for (let i = 0; i < data.length; i++) {
        full_text = data[i];
        text = full_text['text'];
        content_obj.push({ insert: text, attributes: { style: "color:red" } })
    }
    quill.setContents(content_obj);

    content = ""
    accumulated_length = 0;
    for (let i = 0; i < data.length; i++) {
        full_text = data[i];
        text = full_text['text'];
        classname = full_text['classname'];
        score = full_text['score'];
        if (writing_model == "model-writing-1") {
            data_tooltips = "S" + (i + 1) + ":aspect=" + classname
        } else if (writing_model == "model-writing-2") {
            data_tooltips = "S" + (i + 1) + ":score=" + classname.slice(-1)
        }
        let props = {
            id: i,
            type: classname,
            data_tooltip: data_tooltips,
        }
        quill.formatText(accumulated_length, text.length, 'predict-diversity', props);
        accumulated_length += text.length
        content += "<div class='predict selections' style='display:none' selection-id=" + i + " id=" + i + ">" + text + "</div>";
    }
    $('.selected-sentences').html(content)
}




/*****************************************/
// Switch Writing Model Instructions
/*****************************************/
$(".model-switch-btn").click(function (evt) {
    var target = $(evt.currentTarget);
    var model = target.attr("id");
    $("#writing-model").val(model);
    $(".model-instruction").hide();
    $(`.${model}-instruction`).show();
    if (typeof (diversity_text) != "undefined") {
        if (model === "model-writing-1") {
            render_model_output(diversity_text, "model-writing-1");
        } else {
            render_model_output(quality_text, "model-writing-2");
        }
    }
});





/*****************************************/
// Double Click to Select Sentences.
/*****************************************/

function create_container(type, counter) {
    const $ret = $('<div />')
        .addClass('container-box')
        .data({ type, counter })
        .append(
            $('<span />')
                .addClass('remove')
                .append('&times;')
        );
    return $ret
}



/*****************************************/
// Three Predefined Writing Examples
/*****************************************/
var samples = document.getElementById("abstract-sample");
samples.addEventListener("change", function () {
    if (samples.value == "1") {
        // [1] Shen, Hua, and Ting-Hao'Kenneth Huang. "Explaining the Road Not Taken." HCXAI Workshop, CHI (2021).
        quill.setContents([])
        quill.setContents([
            { insert: 'Existing self-explaining models typically favor extracting the shortest possible rationales — snippets of an input text “ responsible for ” corresponding output — to explain the model prediction , with the assumption that shorter rationales are more intuitive to humans . ' },
            { insert: 'However , this assumption has yet to be validated . ' },
            { insert: 'To answer this question , we design a self-explaining model , LimitedInk , which allows users to extract rationales at any target length . ' },
            { insert: 'Compared to existing baselines , LimitedInk achieves compatible endtask performance and human-annotated rationale agreement , making it a suitable representation of the recent class of self-explaining models . ' },
            { insert: 'We use LimitedInk to conduct a user study on the impact of rationale length , where we ask human judges to predict the sentiment label of documents based only on LimitedInk generated rationales with different lengths . ' },
            { insert: 'We show rationales that are too short do not help humans predict labels better than randomly masked text , suggesting the need for more careful design of the best human rationales .' },
        ]);
    }
    if (samples.value == "2") {
        // [2] Shen, Hua, Tongshuang Wu, Wenbo Guo, and Ting-Hao'Kenneth Huang. "Are Shortest Rationales the Best Explanations for Human Understanding?." ACL, 2022.
        quill.setContents([])
        quill.setContents([
            { insert: 'It is unclear if existing interpretations of deep neural network models respond effectively to the needs of users. ' },
            { insert: 'This paper summarizes the common forms of explanations (such as feature attribution, decision rules, or probes) used in over 200 recent papers about natural language processing (NLP), and compares them against user questions collected in the XAI Question Bank. ' },
            { insert: 'We found that although users are interested in explanations for the road not taken — namely, why the model chose one result and not a well-defined, seemly similar legitimate counterpart — most model interpretations cannot answer these questions. ' },
        ]);
    }
    if (samples.value == "3") {
        // [3] Zhang, Xinyang, Ningfei Wang, Hua Shen, Shouling Ji, Xiapu Luo, and Ting Wang. "Interpretable deep learning under fire." USENIX Security, 2020.
        quill.setContents([])
        quill.setContents([
            { insert: 'Providing explanations for deep neural network (DNN) models is crucial for their use in security-sensitive domains. ' },
            { insert: 'A plethora of interpretation models have been proposed to help users understand the inner workings of DNNs: how does a DNN arrive at a specific decision for a given input? ' },
            { insert: 'The improved interpretability is believed to offer a sense of security by involving human in the decision-making process. ' },
            { insert: 'Yet, due to its data-driven nature, the interpretability itself is potentially susceptible to malicious manipulations, about which little is known thus far. ' },
            { insert: 'Here we bridge this gap by conducting the first systematic study on the security of interpretable deep learning systems (IDLSes). ' },
            { insert: 'We show that existing IDLSes are highly vulnerable to adversarial manipulations. ' },
            { insert: 'Specifically, we present ADV2, a new class of attacks that generate adversarial inputs not only misleading target DNNs but also deceiving their coupled interpretation models. ' },
            { insert: "Through empirical evaluation against four major types of IDLSes on benchmark datasets and in security-critical applications (e.g., skin cancer diagnosis), we demonstrate that with ADV2 the adversary is able to arbitrarily designate an input's prediction and interpretation. " },
            { insert: 'Further, with both analytical and empirical evidence, we identify the prediction-interpretation gap as one root cause of this vulnerability - a DNN and its interpretation model are often misaligned, resulting in the possibility of exploiting both models simultaneously. ' },
            { insert: 'Finally, we explore potential countermeasures against ADV2, including leveraging its low transferability and incorporating it in an adversarial training framework. ' },
            { insert: 'Our findings shed light on designing and operating IDLSes in a more secure and informative fashion, leading to several promising research directions. ' },
        ]);
    }
});
