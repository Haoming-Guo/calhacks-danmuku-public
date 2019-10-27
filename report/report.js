const { reactiveProp } = VueChartJs.mixins

Vue.component('emotions', {
    extends: VueChartJs.Radar,
    mixins: [reactiveProp],
    props: ['options'],
    mounted() {
        this.renderChart(this.chartData, this.options)
    }
})

Vue.component('density', {
    extends: VueChartJs.Bar,
    props: ['chartData', 'options'],
    mounted() {
        this.renderChart(this.chartData, this.options)
    }
})

var app = new Vue({
    el: '#app',
    data: {
        vid: data.vid,
        player: null,
        intervals: data.intervals,
        length: 0,
        position: 2,
        currIndex: 0,
        currEmotion: -1,
        selectedEmotion: null,
        emotionData: null,
        emotionOptions: null,
        densityData: null,
        densityOptions: null,
        sumData: null,
        sumOptions: null
    },
    created: function () {
        //this.selectEmotions(4)
        this.updateEmotions(0)
        this.emotionOptions = {
            'scale': {
                'ticks': {
                    suggestedMin: 0,
                    suggestedMax: 0.6
                }
            },
            'responsive': true
        }

        this.densityData = {
            'labels': data.histogram[0],
            'datasets': [{
                'label': 'Number of Comments',
                'backgroundColor':'rgba(54, 162, 235)',
                'borderColor':'rgb(54, 162, 235)',
                'pointBackgroundColor':'rgb(54, 162, 235)',
                'data': data.histogram[1]
            }]
        }
        this.sumData = {
            'labels': data.histogram[0],
            'datasets': [{
                'label': 'Sum of Sentiment Points',
                'backgroundColor':'rgb(75, 192, 192)',
                'borderColor':'rgb(75, 192, 192)',
                'pointBackgroundColor':'rgb(75, 192, 192)',
                'data': data.histogram[2]
            }]
        }
        this.densityOptions = {
            scales: {
                yAxes: [{
                  ticks: {
                    beginAtZero:true
                  }
                }]
              }
        }
        this.sumOptions = this.densityOptions

        for (var i of this.intervals) {
            var l = i[1] - i[0]
            this.length += l
        }
        this.loadYoutubeAPI()
    },
    methods: {
        getProgress: function (index, pos) {
            if (index < this.currIndex) {
                return 100
            } else if (index > this.currIndex) {
                return 0
            } else {
                var interval = this.intervals[index]
                return 100 * (pos - interval[0]) / (interval[1] - interval[0])
            }
        },
        selectEmotions: function (s) {
            s = Math.min(data.emotion_data.length - 1, parseInt(s))
            this.selectedEmotion = {
                'label': 'Selected',
                'backgroundColor':'rgba(54, 162, 235, 0.2)',
                'borderColor':'rgb(54, 162, 235)',
                'pointBackgroundColor':'rgb(54, 162, 235)',
                'data': data.emotion_data[s]
            }
            this.updateEmotions(this.currEmotion, true)
        },
        updateEmotions: function (s, force=false) {
            s = Math.min(data.emotion_data.length - 1, parseInt(s))
            if (force || s != this.currEmotion) {
                this.emotionData = {
                    'labels': data.emotion_types,
                    'datasets': [{
                        'label': 'Emotions',
                        'backgroundColor':'rgba(255, 99, 132, 0.2)',
                        'borderColor':'rgb(255, 99, 132)',
                        'pointBackgroundColor':'rgb(255, 99, 132)',
                        'data': data.emotion_data[s]
                    }]//, this.selectedEmotion]
                }
                this.currEmotion = s
            }
        },
        updatePosition: function () {
            this.position = this.player.getCurrentTime()
            this.updateEmotions(this.position)
            if (this.currIndex < this.intervals.length) {
                var interval = this.intervals[this.currIndex]
                if (this.player.getPlayerState() == 1 && this.position > interval[1]) {
                    this.playInterval(this.currIndex + 1)
                }
            } else {
                if (this.player.getPlayerState() == 1) {
                    this.player.pauseVideo()
                }
            }
        },
        loadYoutubeAPI: function () {
            var tag = document.createElement('script')
            tag.src = 'https://www.youtube.com/iframe_api'
            var firstScriptTag = document.getElementsByTagName('script')[0]
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag)
        },
        onYouTubeIframeAPIReady: function () {
            this.player = new YT.Player('v-1', {
                height: '608',
                width: '1080',
                videoId: this.vid,
                playerVars: {
                    'controls': 0,
                    'disablekb': 1,
                    'modestbranding': 1,
                    'rel': 0,
                    'showinfo': 0,
                    'autohide': 1
                },
                events: {
                    'onReady': this.onPlayerReady
                }
            })
        },
        onPlayerReady: function (event) {
            this.playInterval(0)
            setInterval(this.updatePosition, 30)
        },
        playInterval: function (index) {
            if (index < this.intervals.length) {
                var interval = this.intervals[index]
                this.player.seekTo(interval[0], true)
                this.position = interval[0]
                this.player.playVideo()
            } else {
                this.player.pauseVideo()
            }
            this.currIndex = index
        }
    }
})

onYouTubeIframeAPIReady = app.onYouTubeIframeAPIReady