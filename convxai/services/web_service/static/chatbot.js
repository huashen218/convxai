/**
 * @file This source code supports the user interface of the ConvXAI system.
 * @copyright Hua Shen, Ruchi Panchanadikar, 2022
**/


var target_conference_selection = document.getElementById("target-conference");

function initiateConversation() {
    // ********** Initialize the Conversations and show the Reviews ********** //
    chatBotSession.innerHTML = ""
    typeOfContainer = "init"
    createContainer(typeOfContainer)
    setTimeout(function () {
    }, 1)
    message = JSON.stringify({
        "writingInput": [""],
        "explainInput": target_conference_selection.value,
        "writingIndex": [0],
        "ButtonID": "None"
    })
    socket.emit(
        "interact",
        {
            "text": message,
            "message_type": "conv",
            "writing_model": 'model-writing-1',
            "conversation_id": $("#conversation_id").val(),
        }
    );
}




/*****************************************/
// Initialize the Conversations
/*****************************************/
var chatBotBlankMessageReply = "Type something?"
var chatBotReply = ""
var inputMessage = ""
var typeOfContainer = ""
var ButtonID = ""
var chatBotInitiateMessage = "Hi there, I am <strong>ConvXAI</strong> 👩🏻‍🎓, an <strong>AI Explainer Assistant</strong> to help improve your scientific writing! I'm <strong>generating a review</strong> for your reference. Please wait..."
var chatBotInitiateButtonMessage = "I have seen over <strong>9335 scientific paper abstracts</strong> in ACL, CHI and ICLR conferences published from 2018 to 2022. I'd love to provide some basic feedback to <strong>help improve your abstract writing</strong>. If needed, please share with me which conference you are most interested in?"
var chatBotAskForXAIButtonMessage = ""
var chatBotInitXAIButtonMessage = ""
var chatBotGlobalXAIButtonMessage = ""
var chatBotLocalXAIButtonMessage = ""
var chatBotInstanceExplanation = ""
var chatBotGoodInstanceExplanation = ""
var checkResponse = ""
var chatBotSession = document.querySelector(".chatBot .chatBody .chatSession")
var chatBotTextArea = document.querySelector(".chatBot .chatForm #chatTextBox")
var chatBotSendButton = document.querySelector(".chatBot .chatForm #sendButton")
var instance_ButtonID = ["[InstanceWhy]", "[Prediction Confidence]", "[Similar Examples]", "[Important Words]", "[Counterfactual Explanation]"]
var userHasScrolled = false;

function validateMessage() {
    // ********** Validate there're contents in the TextArea ********** //
    if (chatBotTextArea.value == "" || chatBotTextArea.value == "Type here...")
        return false
    else
        return true
}



chatBotTextArea.addEventListener("keyup", function (event) {
    // ********** This function enables "Enter" Key to send messages ********** //
    if (event.keyCode === 13) {
        event.preventDefault();
        chatBotSendButton.click();
    }
});



chatBotSendButton.addEventListener("click", (event) => {
    // ********** Chatbot Main Functions ********** //
    event.preventDefault()
    inputMessage = chatBotTextArea.value
    var selected_ids = [];
    $(".selections.predict-selected").each(function (index, element) {
        console.log("!!!!element", element);
        selected_ids.push($(element).attr("id"));
    });
    if (instance_ButtonID.includes(ButtonID)) {
        if (selected_ids.length == 0) {
            chatBotReply = "Please <strong>click</strong> one sentence to be explained."
            var typeOfContainer = "reply"
            createContainer(typeOfContainer)
            return;
        }
    }
    var selected_texts = [];
    selected_ids.forEach(function (item, index) {
        selected_texts.push($('.selections.predict-selected').filter('[id=' + item + ']').text());
    });
    message = JSON.stringify({
        "writingInput": selected_texts,
        "explainInput": inputMessage,
        "writingIndex": selected_ids,
        "ButtonID": ButtonID,
    })
    typeOfContainer = "message"
    createContainer(typeOfContainer)
    typeOfContainer = "wait"
    createContainer(typeOfContainer)
    socket.emit(
        "interact",
        {
            "text": message,
            "message_type": "conv",
            "writing_model": 'model-writing-1',
            "conversation_id": $("#conversation_id").val(),
        }
    );
    chatBotTextArea.value = ""
    selected_ids = []
    chatBotTextArea.focus();
    chatBotSession.scrollTop = chatBotSession.scrollHeight;    //Scroll handling for button clicked from chat history
    userHasScrolled = false;
    return;

})




socket.on("conv_response", function (data) {
    // ********** Receive messages from server. ********** //
    var allwaitAnimates = document.getElementsByName("wait-animate")   // remove the typing...
    if (allwaitAnimates.length != 0) {
        var lastwaitAnimates = allwaitAnimates[allwaitAnimates.length - 1]
        lastwaitAnimates.style.display = "none";
    }
    writingIndex = data.writingIndex     // get sentence index of user's selection
    responseIndicator = data.responseIndicator
    if (writingIndex == 1) {
        chatBotReplys = data.text
        for (let i = 0; i < chatBotReplys.length; i += 1) {
            chatBotReply = chatBotReplys[i]
            var typeOfContainer = "reply"
            createContainer(typeOfContainer)
        }
    }
    else if (writingIndex == 2) {  //* return Review Comments -- ["ACL", "CHI", "ICLR"] */
        chatBotReplys = data.text
        chatBotReply = chatBotReplys[0]
        var typeOfContainer = "reply"
        createContainer(typeOfContainer)
        chatBotAskForXAIButtonMessage = chatBotReplys[1]
        var typeOfContainer = "ask-for-xai"
        createContainer(typeOfContainer)
    }

    else if (writingIndex == 3) {   //* return [GlobalWhy] */
        chatBotReplys = data.text
        chatBotGlobalXAIButtonMessage = chatBotReplys[0]
        var typeOfContainer = "global-explanation"
        createContainer(typeOfContainer)
    }

    else if (writingIndex == 4) {   //* return [InstanceWhy] */
        chatBotReplys = data.text
        for (let i = 0; i < chatBotReplys.length - 1; i += 2) {
            chatBotInstanceExplanation = chatBotReplys[i + 1]
            var typeOfContainer = chatBotReplys[i]
            createContainer(typeOfContainer)
        }
    }

    else if (writingIndex == 6) {   //* check variables */
        chatBotReplysAll = data.text
        chatBotReplys = chatBotReplysAll[0]
        for (let i = 0; i < chatBotReplys.length; i += 1) {
            chatBotReply = chatBotReplys[i]
            var typeOfContainer = "reply"
            createContainer(typeOfContainer)
        }
        chatBotReply = chatBotReplysAll[1]
        var typeOfContainer = "reply"
        createContainer(typeOfContainer)
    }
})


function createContainer(typeOfContainer) {
    var containerID = ""
    var textClass = ""
    switch (typeOfContainer) {
        case "init":
            containerID = "replyContainer"
            textClass = "init"
            break;
        case "ask-for-xai":
            containerID = "replyContainer"
            textClass = "ask-for-xai"
            break;
        case "global-explanation":
            containerID = "replyContainer"
            textClass = "global-explanation"
            break;
        case "local-explanation":
            containerID = "replyContainer"
            textClass = "local-explanation"
            break;
        case "message":
            containerID = "messageContainer"
            textClass = "message"
            break;
        case "wait":
            containerID = "replyContainer"
            textClass = "wait"
            break;
        case "reply":
            containerID = "replyContainer"
            textClass = "reply"
            break;
        case "error":
            containerID = "replyContainer"
            textClass = "reply"
            break;
        case "long":
            containerID = "replyContainer"
            textClass = "long"
            break;
        case "short":
            containerID = "replyContainer"
            textClass = "short"
            break;
        case "quality":
            containerID = "replyContainer"
            textClass = "quality"
            break;
        case "aspect":
            containerID = "replyContainer"
            textClass = "aspect"
            break;
        default:
            alert("Error! Please reload the webiste.")
    }


    // ********** Creating container ********** //
    var newContainer = document.createElement("div")
    newContainer.setAttribute("class", "container")
    if (containerID == "messageContainer")
        newContainer.setAttribute("id", "messageContainer")
    if (containerID == "replyContainer")
        newContainer.setAttribute("id", "replyContainer")
    chatBotSession.appendChild(newContainer)


    switch (textClass) {
        case "init":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotInitiateMessage
            lastReplyContainer.appendChild(newReply)
            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "message":
            var allMessageContainers = document.querySelectorAll("#messageContainer")
            var lastMessageContainer = allMessageContainers[allMessageContainers.length - 1]
            var newMessage = document.createElement("p")
            newMessage.setAttribute("class", "message animateChat")
            newMessage.innerHTML = inputMessage
            lastMessageContainer.appendChild(newMessage)
            lastMessageContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            chatBotSession.scrollTop = chatBotSession.scrollHeight;
            break


        case "wait":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.setAttribute("name", "wait-animate")
            switch (typeOfContainer) {
                case "wait":
                    newReply.innerHTML = "Typing..."
                    break
                default:
                    newReply.innerHTML = "Sorry! I could not understannd."
            }
            lastReplyContainer.appendChild(newReply)
            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "reply":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            switch (typeOfContainer) {
                case "reply":
                    newReply.innerHTML = chatBotReply
                    break
                case "error":
                    newReply.innerHTML = chatBotBlankMessageReply
                    break
                default:
                    newReply.innerHTML = "Sorry! I could not understannd."
            }
            lastReplyContainer.appendChild(newReply)
            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "ask-for-xai":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotAskForXAIButtonMessage
            var breakline = document.createElement("br")
            var btnNO = document.createElement("button");
            btnNO.setAttribute("class", "reply xaiquestionButton")
            btnNO.setAttribute("type", "button")
            btnNO.setAttribute("id", "[InstanceWhy]")
            btnNO.innerHTML = "How should I improve?"
            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(breakline)
            newReply.appendChild(btnNO)
            btnNO.onclick = handleExplanationButtonClicked;
            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "global-explanation":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotGlobalXAIButtonMessage

            var btnData = document.createElement("button");
            btnData.setAttribute("class", "reply xaiquestionButton")
            btnData.setAttribute("type", "button")
            btnData.setAttribute("id", "[Data Statistics]")
            btnData.innerHTML = "Data Statistics"

            var btnModel = document.createElement("button");
            btnModel.setAttribute("class", "reply xaiquestionButton")
            btnModel.setAttribute("type", "button")
            btnModel.setAttribute("id", "[Model Description]")
            btnModel.innerHTML = "Model Description"

            var qualityScore = document.createElement("button");
            qualityScore.setAttribute("class", "reply xaiquestionButton")
            qualityScore.setAttribute("type", "button")
            qualityScore.setAttribute("id", "[Quality Score Range]")
            qualityScore.innerHTML = "Quality Score Range"

            var aspectLabel = document.createElement("button");
            aspectLabel.setAttribute("class", "reply xaiquestionButton")
            aspectLabel.setAttribute("type", "button") //
            aspectLabel.setAttribute("id", "[Aspect Distribution]") //
            aspectLabel.innerHTML = "Label Distribution"

            var btnLength = document.createElement("button");
            btnLength.setAttribute("class", "reply xaiquestionButton")  //acl
            btnLength.setAttribute("type", "button") //
            btnLength.setAttribute("id", "[Sentence Length]") //
            btnLength.innerHTML = "Sentence Length"

            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(btnData)
            newReply.appendChild(btnModel)
            newReply.appendChild(qualityScore)
            newReply.appendChild(aspectLabel)
            newReply.appendChild(btnLength)

            btnData.onclick = handleExplanationButtonClicked;
            btnModel.onclick = handleExplanationButtonClicked;
            qualityScore.onclick = handleExplanationButtonClicked;
            aspectLabel.onclick = handleExplanationButtonClicked;
            btnLength.onclick = handleExplanationButtonClicked;

            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "local-explanation":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotLocalXAIButtonMessage

            var btnConfidence = document.createElement("button");
            btnConfidence.setAttribute("class", "reply xaiquestionButton")
            btnConfidence.setAttribute("type", "button")
            btnConfidence.setAttribute("id", "[Prediction Confidence]")
            btnConfidence.innerHTML = "Prediction Confidence"

            var btnExample = document.createElement("button");
            btnExample.setAttribute("class", "reply xaiquestionButton")
            btnExample.setAttribute("type", "button")
            btnExample.setAttribute("id", "[Similar Examples]")
            btnExample.innerHTML = "Similar Published Sentences"

            var btnAttribution = document.createElement("button");
            btnAttribution.setAttribute("class", "reply xaiquestionButton")
            btnAttribution.setAttribute("type", "button")
            btnAttribution.setAttribute("id", "[Important Words]")
            btnAttribution.innerHTML = "Important Words"

            var btnCounterfactual = document.createElement("button");
            btnCounterfactual.setAttribute("class", "reply xaiquestionButton")
            btnCounterfactual.setAttribute("type", "button")
            btnCounterfactual.setAttribute("id", "[Counterfactual Explanation]")
            btnCounterfactual.innerHTML = "Counterfacetual examples"

            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(btnConfidence)
            newReply.appendChild(btnExample)
            newReply.appendChild(btnAttribution)
            newReply.appendChild(btnCounterfactual)

            btnConfidence.onclick = handleExplanationButtonClicked;
            btnExample.onclick = handleExplanationButtonClicked;
            btnAttribution.onclick = handleExplanationButtonClicked;
            btnCounterfactual.onclick = handleExplanationButtonClicked;

            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "long":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotInstanceExplanation


            var btnLength = document.createElement("button");
            btnLength.setAttribute("class", "reply xaiquestionButton")
            btnLength.setAttribute("type", "button") //
            btnLength.setAttribute("id", "[Sentence Length]") //
            btnLength.innerHTML = "Sentence Length"


            var btnExample = document.createElement("button");
            btnExample.setAttribute("class", "reply xaiquestionButton")
            btnExample.setAttribute("type", "button") //
            btnExample.setAttribute("id", "[Similar Examples]") //
            btnExample.innerHTML = "Similar Published Sentences"


            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(btnLength)
            newReply.appendChild(btnExample)
            btnLength.onclick = handleExplanationButtonClicked;
            btnExample.onclick = handleExplanationButtonClicked;
            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "short":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotInstanceExplanation

            var btnLength = document.createElement("button");
            btnLength.setAttribute("class", "reply xaiquestionButton")
            btnLength.setAttribute("type", "button") //
            btnLength.setAttribute("id", "[Sentence Length]") //
            btnLength.innerHTML = "Sentence Length"

            var btnExample = document.createElement("button");
            btnExample.setAttribute("class", "reply xaiquestionButton")
            btnExample.setAttribute("type", "button") //
            btnExample.setAttribute("id", "[Similar Examples]") //
            btnExample.innerHTML = "Similar Published Sentences"


            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(btnLength)
            newReply.appendChild(btnExample)
            btnLength.onclick = handleExplanationButtonClicked;
            btnExample.onclick = handleExplanationButtonClicked;

            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })

            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)

            break


        case "quality":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotInstanceExplanation
            var breakline = document.createElement("br")

            var qualityScore = document.createElement("button");
            qualityScore.setAttribute("class", "reply xaiquestionButton")
            qualityScore.setAttribute("type", "button")
            qualityScore.setAttribute("id", "[Quality Score Range]")
            qualityScore.innerHTML = "Style Quality Score"

            var btnExample = document.createElement("button");
            btnExample.setAttribute("class", "reply xaiquestionButton")
            btnExample.setAttribute("type", "button")
            btnExample.setAttribute("id", "[Similar Examples]")
            btnExample.innerHTML = "Similar Published Sentences"

            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(qualityScore)
            newReply.appendChild(btnExample)
            qualityScore.onclick = handleExplanationButtonClicked;
            btnExample.onclick = handleExplanationButtonClicked;

            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break


        case "aspect":
            var allReplyContainers = document.querySelectorAll("#replyContainer")
            var lastReplyContainer = allReplyContainers[allReplyContainers.length - 1]
            var newReply = document.createElement("p")
            newReply.setAttribute("class", "reply animateChat accentColor")
            newReply.innerHTML = chatBotInstanceExplanation

            var breakline = document.createElement("br")

            var aspectLabel = document.createElement("button");
            aspectLabel.setAttribute("class", "reply xaiquestionButton")
            aspectLabel.setAttribute("type", "button")
            aspectLabel.setAttribute("id", "[Aspect Distribution]")
            aspectLabel.innerHTML = "Label Distribution"

            var btnConfidence = document.createElement("button");
            btnConfidence.setAttribute("class", "reply xaiquestionButton")
            btnConfidence.setAttribute("type", "button")
            btnConfidence.setAttribute("id", "[Prediction Confidence]")
            btnConfidence.innerHTML = "Prediction Confidence"

            var btnExample = document.createElement("button");
            btnExample.setAttribute("class", "reply xaiquestionButton")
            btnExample.setAttribute("type", "button")
            btnExample.setAttribute("id", "[Similar Examples]")
            btnExample.innerHTML = "Similar Published Sentences"

            var btnAttribution = document.createElement("button");
            btnAttribution.setAttribute("class", "reply xaiquestionButton")
            btnAttribution.setAttribute("type", "button")
            btnAttribution.setAttribute("id", "[Important Words]")
            btnAttribution.innerHTML = "Which words are most important for this prediction?"

            var btnCounterfactual = document.createElement("button");
            btnCounterfactual.setAttribute("class", "reply xaiquestionButton")
            btnCounterfactual.setAttribute("type", "button")
            btnCounterfactual.setAttribute("id", "[Counterfactual Explanation]")
            btnCounterfactual.innerHTML = "How can I revise the sentence to get a different label?"


            lastReplyContainer.appendChild(newReply)
            newReply.appendChild(aspectLabel)
            newReply.appendChild(btnConfidence)
            newReply.appendChild(btnExample)
            newReply.appendChild(btnAttribution)
            newReply.appendChild(btnCounterfactual)

            aspectLabel.onclick = handleExplanationButtonClicked;
            btnConfidence.onclick = handleExplanationButtonClicked;
            btnExample.onclick = handleExplanationButtonClicked;
            btnAttribution.onclick = handleExplanationButtonClicked;
            btnCounterfactual.onclick = handleExplanationButtonClicked;

            lastReplyContainer.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
            setTimeout(function () {
                chatBotSession.scrollTop = chatBotSession.scrollHeight;
            }, 50)
            break
        default:
            console.log("Error in conversation")
    }
}




/*****************************************/
// Explanation Buttons Click Function
/*****************************************/

function handleExplanationButtonClicked(e) {
    e.preventDefault();
    ButtonID = e.target.id
    chatBotTextArea = document.querySelector(".chatBot .chatForm #chatTextBox")
    chatBotTextArea.value = ""

    if (ButtonID == "[Data Statistics]") {
        chatBotTextArea.value = "What data did the system learn from?";
    }
    else if (ButtonID == "[Model Description]") {
        chatBotTextArea.value = "What kind of models are used?";
    }
    else if (ButtonID == "[Quality Score Range]") {
        chatBotTextArea.value = "What's the range of the style quality scores?";
    }
    else if (ButtonID == "[Aspect Distribution]") {
        chatBotTextArea.value = "How are the structure labels distributed?";
    }
    else if (ButtonID == "[Sentence Length]") {
        chatBotTextArea.value = "What's the statistics of the sentence lengths?";
    }
    else if (ButtonID == "[Prediction Confidence]") {
        chatBotTextArea.value = "How confident is the model for this prediction?";
    }
    else if (ButtonID == "[Similar Examples]") {
        chatBotTextArea.value = "What are some published sentences that look similar to mine semantically?"
    }
    else if (ButtonID == "[Important Words]") {
        chatBotTextArea.value = "Which words in this sentence are most important for this prediction?";
    }
    else if (ButtonID == "[Counterfactual Explanation]") {
        chatBotTextArea.value = "How can I revise the input to get a different prediction label?";
    }
    else if (ButtonID == "[GlobalWhy]") {
        chatBotTextArea.value = "Can you show me model and data information in general?";
    }
    else if (ButtonID == "[InstanceWhy]") {
        chatBotTextArea.value = "How to improve the sentence?";
    }
    else {
        console.log("There is an Error here!")
    }
    return;
}



/*****************************************/
// Help Icon Functions
/*****************************************/

$(".help-icon").click(function () {
    $(document).find("#" + this.id + "-text").toggle()
})

$("#sentence-select-help-text").hover(function () {
    $(document).find("#" + this.id + "-text").display = "block";
})

$(".select-sentence-hover").clkic(function () {
    $(document).find("#sentence-select-help-text").toggle()
})



/*****************************************/
//  Chat Scrolling to the bottom
/*****************************************/
//** Note: The handling for button clicked in chat history is handled in the click listener for the chat send button */

window.setInterval(() => {
    if (!userHasScrolled) {
        chatBotSession.scrollTop = chatBotSession.scrollHeight; 
    }
}, 300)

chatBotSession.addEventListener("wheel", (event) => {                                            //listening for user scroll
    userHasScrolled = true;
    if (chatBotSession.offsetHeight + chatBotSession.scrollTop >= chatBotSession.scrollHeight) { //checking if element is scrolled to the bottom
        userHasScrolled = false;
    }
})




/*****************************************/
//  Click to fill in the question
/*****************************************/
$(".data-question").on('click', function (event) {
    event.stopPropagation();
    event.stopImmediatePropagation();
    chatBotTextArea = document.querySelector(".chatBot .chatForm #chatTextBox")
    chatBotTextArea.value = "What data did the system learn from?";
});