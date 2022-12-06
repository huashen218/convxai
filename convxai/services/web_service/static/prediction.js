
/**
 * @file This source code supports the user interface of the ConvXAI system.
 * @copyright Hua Shen 2022
**/


var BlockEmbed = Quill.import('blots/block/embed');
let Inline = Quill.import('blots/inline');

class PredictDiversity extends Inline {
    static create(props) {
        let node = super.create();
        node.setAttribute('id', props.id);
        node.setAttribute('type', "predict-" + props.type);
        node.setAttribute('category', "predicts");
        node.setAttribute('data_tooltip', props.data_tooltip);
        return node;
    }
    static formats(node) {
        return {
            id: node.getAttribute("id"),
            type: node.getAttribute("type"),
        }
    }
}


PredictDiversity.blotName = 'predict-diversity';
PredictDiversity.tagName = 'p';
PredictDiversity.className = 'predict-model-writing-1';
Quill.register(PredictDiversity);