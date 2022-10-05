//const socket = io.connect('/connection', {'path': '/writing-support-socket.io'});
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
document.getElementById("start").addEventListener("submit", function(event){
    event.preventDefault()
    fetch('/init_conversation', {
        headers: {
            'Content-Type': 'application/json'
        },
        method: 'POST',
        body: JSON.stringify({text: "start"}),
    }).then(response=>response.json()).then(data=>{
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
target_conference_selection.addEventListener("change", function() {
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
            [{list: 'ordered'}, {list: 'bullet'}]
        ]
    },
    toolbar: '#toolbar',
    scrollingContainer: '#editor', 
    placeholder: 'Compose an abstract...'
});






/*****************************************/
// Submit Writing and Update Visualization
/*****************************************/
document.getElementById("editor-form").addEventListener("submit", function(event){

    event.preventDefault();

    // Adding waiting logo;
    $('.btn-submit-loader').css({"display": "block"});

    // Identify writing model;
    var writing_model = $("#writing-model").val();

    socket.emit(
        "interact",
        {
            "text": quill.getText(),
            "message_type": "task",
            "writing_model": writing_model,
            "conversation_id": $("#conversation_id").val(),
        }
    );
    return;

});




/*********************************************************/
// Return Both Models' Predictions and Update Visualization
/*********************************************************/
var diversity_text;
var quality_text;


socket.on("task_response", function(data) {

    // Removing waiting logo;
    $('.btn-submit-loader').css({"display": "none"});
    //$('.writing-model-descriptions').show("slow");
    $('.writing-model-descriptions').slideDown("500");
    
    $('#display-xai').fadeIn(300);


    // Get the model predictions -- [text, classname, score]; and put them into quill textarea (using quill.setContents function).
    data = data.text;

    diversity_text = data.slice(0, data.length / 2);
    quality_text  = data.slice(data.length / 2, data.length);

    // Default setting: return predictions of model-writing-1.
    var writing_model = $("#writing-model").val();
    if (writing_model === "" || writing_model === "model-writing-1") {
        $("#writing-model").val("model-writing-1");
        // $(`.model-writing-1-instruction`).show();
        $(`.model-writing-1-instruction`).toggle();
        render_model_output(diversity_text, "model-writing-1");
    } else {
        // $(`.model-writing-2-instruction`).show();
        $(`.model-writing-2-instruction`).toggle();
        render_model_output(quality_text, "model-writing-2");
    }
    
    initiateConversation();
}
);







/*****************************************/
// Select Conferences
/*****************************************/





// var chatBotTextArea  = document.querySelector( ".chatBot .chatForm #chatTextBox" )


//  Hover functions.
// $(".model-writing-1").hover( function (e) {
//     id = e.id;
//     $('[category=predicts]').filter('[id='+id+']').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
//     $('.comments').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//     $('.selections').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
// });




$(document).on('click', '[category="predicts"]', function(e) {
    $('[id='+this.id+']').toggleClass("predict-selected");
    $('[selection-id='+this.id+']').toggle();
    $('[category=predicts]').filter('[id='+this.id+']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id='+this.id+']').toggleClass('associate-highlight');
    $('.selections').filter('[id='+this.id+']').toggleClass('associate-highlight');
});





$(document).on('click', '.comments', function(e) {
    $('[id='+this.id+']').toggleClass("predict-selected");
    $('[selection-id='+this.id+']').toggle();
    $('[category=predicts]').filter('[id='+this.id+']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id='+this.id+']').toggleClass('associate-highlight');
    $('.selections').filter('[id='+this.id+']').toggleClass('associate-highlight');
});



$(document).on('click', '.selections', function(e) {
    // $(document).find("#"+this.id.split('-')[1]).removeClass("predict-selected")
    // $(this).css("display","none");
    $('[id='+this.id+']').toggleClass("predict-selected");
    $('[selection-id='+this.id+']').toggle();
    $('[category=predicts]').filter('[id='+this.id+']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id='+this.id+']').toggleClass('associate-highlight');
    $('.selections').filter('[id='+this.id+']').toggleClass('associate-highlight');
});



$(document).on('dblclick', '.comments', function(e) {
// $( ".comments" ).dblclick(function(e) {
    $('[id='+this.id+']').toggleClass("predict-selected");
    $('[selection-id='+this.id+']').toggle();
    $('[category=predicts]').filter('[id='+this.id+']').toggleClass('data_tooltip_associate_highlight');
    $('.comments').filter('[id='+this.id+']').toggleClass('associate-highlight');
    $('.selections').filter('[id='+this.id+']').toggleClass('associate-highlight');
    chatBotTextArea             = document.querySelector( ".chatBot .chatForm #chatTextBox" )
    chatBotTextArea.value = "Can you explain this sentence?";
  });














/*****************************************/
// Double Click to Select Sentences.
/*****************************************/

function render_model_output(data, writing_model) {
    console.log("render_model_outputppp", data);
    var data_tooltips = ""

    const content_obj = [];
    for (let i = 0; i < data.length; i++){
        full_text = data[i];
        text = full_text['text'];
        content_obj.push({insert: text, attributes: {style: "color:red"}})
        // content_obj.push({ insert: '\n' })
        // <div class="ql-tooltip ql-editing" data-mode="link" style="left: 24.2578px; top: 170.594px;"><a class="ql-preview" target="_blank" href="about:blank"></a><input type="text" data-formula="e=mc^2" data-link="https://quilljs.com" data-video="Embed URL" placeholder="https://quilljs.com"><a class="ql-action"></a><a class="ql-remove"></a></div>
    }
    quill.setContents(content_obj);

    content = ""
    // Assign the CSS attributes (id, type) for each sentence (using quill.formatText function)..
    accumulated_length = 0;
    for (let i = 0; i < data.length; i++){
        full_text = data[i];
        text = full_text['text'];
        classname = full_text['classname'];
        score = full_text['score'];

        if (writing_model=="model-writing-1"){
            data_tooltips = "S"+(i+1) + ":aspect=" + classname
        } else if (writing_model=="model-writing-2"){
            data_tooltips = "S"+(i+1) + ":score=" + classname.slice(-1)
        }
        
        let props = {
            id: i,
            type: classname,
            data_tooltip: data_tooltips,
            // data_tooltip: "S"+(i+1) + ": aspect=" + classname + "  |  score=" + score,
        }

        quill.formatText(accumulated_length, text.length, 'predict-diversity', props);
        accumulated_length += text.length
        content += "<div class='predict selections' style='display:none' selection-id=" + i + " id=" + i + ">" + text + "</div>";

    
    }
    // $(".selected-sentences").val(content);
    // $(".selected-sentences").text = content;
    // $(".selected-sentences").append(content);
    $('.selected-sentences').html(content)
}




/*****************************************/
// Switch Writing Model Instructions
/*****************************************/
$(".model-switch-btn").click(function(evt) {
    // identify the correct information
    var target = $(evt.currentTarget);
    var model = target.attr("id");

    // keep the info
    $("#writing-model").val(model);
    
    // hide all instruction & show the correct one
    $(".model-instruction").hide();
    $(`.${model}-instruction`).show();
    // $(`.${model}-instruction`).toggle();

    // hide all predictions & show the correct one
    // $('[category="predicts"]').hide();
    // $(`.predict-${model}`).show();
    if (typeof(diversity_text) != "undefined") {
        if (model === "model-writing-1" ){
            render_model_output(diversity_text, "model-writing-1");
        } else {
            render_model_output(quality_text, "model-writing-2");
        }
    }
});




// ########################################################################







/*****************************************/
// Double Click to Select Sentences.
/*****************************************/

function create_container(type, counter) {
    const $ret = $('<div />')
        .addClass('container-box')
        .data({type, counter})
        .append(
            $('<span />')
                .addClass('remove')
                .append('&times;')
        );
    return $ret
  }







var samples = document.getElementById("abstract-sample");
samples.addEventListener("change", function() {
    console.log("samples.value", samples.value);
    if (samples.value == "1") {
        quill.setContents([])
        quill.setContents([
            { insert: 'Existing self-explaining models typically favor extracting the shortest possible rationales — snippets of an input text “ responsible for ” corresponding output — to explain the model prediction , with the assumption that shorter rationales are more intuitive to humans . ' },
            { insert: 'However , this assumption has yet to be validated . '},
            { insert: 'To answer this question , we design a self-explaining model , LimitedInk , which allows users to extract rationales at any target length . ' },
            { insert: 'Compared to existing baselines , LimitedInk achieves compatible endtask performance and human-annotated rationale agreement , making it a suitable representation of the recent class of self-explaining models . ' },
            { insert: 'We use LimitedInk to conduct a user study on the impact of rationale length , where we ask human judges to predict the sentiment label of documents based only on LimitedInk generated rationales with different lengths . ' },
            { insert: 'We show rationales that are too short do not help humans predict labels better than randomly masked text , suggesting the need for more careful design of the best human rationales .' },
          ]);
    }
    if (samples.value == "2") {
        quill.setContents([])
        quill.setContents([
            { insert: 'It is unclear if existing interpretations of deep neural network models respond effectively to the needs of users. ' },
            { insert: 'This paper summarizes the common forms of explanations (such as feature attribution, decision rules, or probes) used in over 200 recent papers about natural language processing (NLP), and compares them against user questions collected in the XAI Question Bank. '},
            { insert: 'We found that although users are interested in explanations for the road not taken — namely, why the model chose one result and not a well-defined, seemly similar legitimate counterpart — most model interpretations cannot answer these questions. ' },
          ]);
    }
    if (samples.value == "3") {
        quill.setContents([])
        quill.setContents([
            { insert: 'Providing explanations for deep neural network (DNN) models is crucial for their use in security-sensitive domains. ' },
            { insert: 'A plethora of interpretation models have been proposed to help users understand the inner workings of DNNs: how does a DNN arrive at a specific decision for a given input? '},
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








// $(document).on('click', '[category="predicts"]', function(event) {
//     $('[id='+this.id+']').toggleClass("predict-selected");
//     $('[selection-id='+this.id+']').toggle();
// });

// $(document).on('click', ".predict.selections",function(event) {
//     $(document).find("#"+this.id.split('-')[1]).removeClass("predict-selected")
//     $(this).css("display","none");
// });

// $(document).on('click', '.comments', function(event) {
//     $('[id='+this.id+']').toggleClass("predict-selected");
//     $('[selection-id='+this.id+']').toggle();
// });



// $('#abstract-sample').val()




// $("p[category='predicts']").tooltips


// $('.predict-model-writing-1').tooltip('show');





// /*****************************************/
// // Switch Writing Model Instructions
// /*****************************************/
// $(".model-switch-btn").click(function(evt) {
//     // identify the correct information
//     var target = $(evt.currentTarget);
//     var model = target.attr("id");

//     // keep the info
//     $("#writing-model").val(model);
    
//     // hide all instruction & show the correct one
//     $(".model-instruction").hide();
//     $(`.${model}-instruction`).show();

//     // hide all predictions & show the correct one
//     $('[category="predicts"]').hide();
//     $(`.predict-${model}`).show();
// });



// $(function(){
//     $("*[data_tooltip]").hover(
//          function(){
//              $(".model-writing-1").css('border','1px solid transparent');
//              $(".model-writing-1").css('background-color','#EAEAEA');
//          }
//         //  function(){
//         //      $("#innerContainer").css('border-color','#000');
//         //      $("#outerContainer").css('border-color','#000');
//         //  }
//      );
//  });























// $('.model-writing-1').hover(
//     function(){ $('.model-writing-2').addClass('hover') },
//     function(){ $('.model-writing-2').removeClass('hover')}
// );

// $(".model-writing-1").hover( function (e) {
//     $('.model-writing-2').toggleClass('model-writing-2-hover', e.type === 'mouseenter');
// });



// $(".model-writing-1").hover( function (e) {
//     $('[data_tooltip="S1: aspect=background"]').toggleClass('data_tooltip-hover', e.type === 'mouseenter');
// });



// $(".model-writing-1").hover( function (e) {
//     $('.model-writing-2').toggleClass('associate-highlight', e.type === 'mouseenter'); 
//     $('.model-writing-2').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
// });




    // $('.model-writing-2').toggleClass('associate-highlight', e.type === 'mouseenter'); 
    // $('[data_tooltip="S1: aspect=background"]').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
    // $('.comments').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
    

// $('#0').filter('[myid="1"],[myid="2"]');



// $(".model-writing-1").hover( function (e) {
//     $('.model-writing-2').toggleClass('associate-highlight', e.type === 'mouseenter'); 
//     $('.model-writing-2').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');

// });



// $('.model-writing-2').toggleAttribute('data_tooltip_associate_highlight', e.type === 'mouseenter');
// $('.model-writing-2').attr('data_tooltip_associate_highlight', function(_, attr){ return !attr}); 







// document.querySelectorAll('.comments').forEach(item => {
//     item.addEventListener('mouseover', event => {
//         console.log("event id", id);
//         id = event.target.id
//         $('[category=predicts]').filter('[id='+id+']').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
//         $('.comments').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//         $('.selections').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//     })
//   })



// $('.comments').hover( function (e) {
//     console.log("id", id);
//     id = e.id;
//     $('[category=predicts]').filter('[id='+id+']').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
//     $('.comments').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//     $('.selections').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
// });




// $('[category=predicts]').click( function (e) {
//     id = e.id;
//     console.log("id", id);
//     $('[category=predicts]').filter('[id='+id+']').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
//     $('.comments').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//     $('.selections').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
// });


// $(document).ready(function()
//   {
//     $('li.active').hover(
//       function(){ 
//         $(this).children("a").addClass("icon-white"); //Add an active class to the anchor
//       },
//       function() {
//         $(this).children("a").removeClass("icon-white"); //Remove an active class to the anchor
//       }
//    )
//  });
 




// document.querySelectorAll('[category=predicts]').forEach(item => {
//     item.hover( function (e) {
//         id = 0
//         $('[category=predicts]').filter('[id='+id+']').toggleClass('data_tooltip_associate_highlight', e.type === 'mouseenter');
//         $('.comments').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//         $('.selections').filter('[id='+id+']').toggleClass('associate-highlight', e.type === 'mouseenter');
//     });
//   })



// $(document).on('dblclick', '[category="predicts"]', function(event) {
//     $('[id='+this.id+']').toggleClass("predict-selected");
//     $('[selection-id='+this.id+']').toggle();

// });



