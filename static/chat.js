function updateBarsAndValues(idx, position, values, probability) {
    var block = document.getElementById(idx);
    var tokens = block.getElementsByClassName('token');
    for (var i = 0; i < tokens.length; i++) {
        var value = values[i] || 0;
        if (value !== 0) {
            var relative_value = value / Math.max(...values);
            var color_value = parseInt(relative_value * 255);
            var color = 'rgb(0, ' + (255 - color_value) + ', 255)';
            var bar_height = 28 * relative_value;
            var bar = tokens[i].getElementsByClassName('bar')[0];
            var valueDiv = tokens[i].getElementsByClassName('value')[0];
            bar.style.height = bar_height + 'px';
            bar.style.backgroundColor = color;
            valueDiv.style.color = color;
            valueDiv.textContent = value.toFixed(4);
            tokens[i].style.backgroundColor = 'rgba(200, 200, 200, 0.3)';
        }
        else {
            var bar = tokens[i].getElementsByClassName('bar')[0];
            bar.style.height = '0px';
            var valueDiv = tokens[i].getElementsByClassName('value')[0];
            valueDiv.textContent = '';
            tokens[i].style.backgroundColor = 'rgba(200, 200, 200, 0.3)';
        }
    }
    var probability_value = parseInt(probability * 255);
    var background_color = 'rgb(255, ' + (255 - probability_value) + ', ' + (255 - probability_value) + ')';
    tokens[position].style.backgroundColor = background_color;
    tokens[position].getElementsByClassName('value')[0].textContent = (probability * 100).toFixed(2) + '%';
    tokens[position].getElementsByClassName('value')[0].style.color = 'black';
}

function reset() {
    var tokens = document.getElementsByClassName('token');
    for (var i = 0; i < tokens.length; i++) {
        var bar = tokens[i].getElementsByClassName('bar')[0];
        bar.style.height = '0px';
        var valueDiv = tokens[i].getElementsByClassName('value')[0];
        valueDiv.textContent = '';
        tokens[i].style.backgroundColor = 'rgba(200, 200, 200, 0.3)';
    }
}

function chatBot() {
    return {
        botTyping: false,
        messages: [{
            from: 'bot',
            text: 'How may I be of service?',
            show: false,
            more: false,
            saliency: "",
            modules: []
        }],
        mediaRecorder: null,
        audioChunks: [],
        isRecording: false,
        init: function() {

        },
        clearChat: function() {
            this.messages= [{
                from: 'bot',
                text: 'How may I be of service?',
                show: false,
                more: false,
                saliency: "",
                modules: []
            }]
        },
        loadChat: function() {
            this.clearChat();

            let self = this;
            $.ajax({
                type: 'GET',
                url: '/load',
                success: function(response) {
                    self.messages = self.messages.concat(response);
                    self.scrollChat();
                },
                error: function(error) {
                    console.error(error);
                }
            });
        },
        output: function(input) {
            this.messages.push({
                from: 'user',
                text: input,
                show: false,
                more: false,
                saliency: "",
                modules: []
            });

            this.scrollChat();
            this.botTyping = true;
            this.scrollChat();

            let self = this;
            $.ajax({
                type: 'POST',
                url: '/query/generation',
                data: { text: input },
                success: function(response) {
                    self.botTyping = false;
                    if ("error" in response) {
                        self.messages.push({
                            from: 'bot',
                            text: response["error"],
                            show: false,
                            more: false,
                            saliency: "",
                            modules: []
                        });
                    }
                    else if ("output" in response) {
                        self.messages.push({
                            from: 'bot',
                            text: response["output"],
                            show: false,
                            more: true,
                            saliency: response["saliency"],
                            modules: response["modules"]
                        });
                    }
                    self.scrollChat();
                },
                error: function(error) {
                    console.error(error);
                    self.botTyping = false;
                    self.messages.push({
                        from: 'bot',
                        text: 'Sorry, an error occured. Please try again.',
                        show: false,
                        more: false,
                        saliency: "",
                        modules: []
                    });
                    self.scrollChat();
                }
            });
        },
        scrollChat: function() {
            const messagesContainer = document.getElementById("messages");
            messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;
            setTimeout(() => {
                messagesContainer.scrollTop = messagesContainer.scrollHeight - messagesContainer.clientHeight;
            }, 100);
        },
        updateChat: function(target) {
            if (target.value.trim()) {
                this.output(target.value.trim());
                target.value = '';
            }
        },
    }
}