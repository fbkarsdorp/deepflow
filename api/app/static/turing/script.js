var censored = [];
axios.get('/static/censuur.json').then(function(res){
  censored = res.data.words;
})

Vue.component('censored', {
  props: ['text','loading','censored'],
  computed: {
    newtext: function (val) {
      var words = this.text.split(' ');
      words.forEach(function(v,k){
        if (censored) {
          censored.forEach(function(vv,kk){
            //var regex = new RegExp('([\'":;\\.,\\-+`?!$%&]{2})?' + vv + '([\'":;\\.,\\-+`?!$%&]{2})?', 'i')
            //if (v.match(regex)) words[k] = '**********************************'.substr(0, v.length)
            if (vv === v) words[k] = '**********************************'.substr(0, v.length)
          })
        }
      })
      return words.join(' ')
    }
  },
  template: '<div>{{censored ? newtext : text}}</div>'
})

Vue.component('scoreboard', {
  props: ['scoreboard','name','score'],
  data () {
    return {
      data: [],
      interval: null
    }
  },
  computed: {
  },
  methods: {
    load () {
      let self = this
      axios.get('/scoreboard').then(function(res){
        if(res.data.status === 'OK') {
          self.data = res.data.ranking
        }
      }).catch(function(err){
        // console.log('error receiving scoreboard', err)
      })
    }
  },
  template: '<div id="scoreboard"><span v-if="name">Your score is {{score}}<br><br></span><label>high score</label><div id="participant" v-for="item in data" :class="{\'active\':item.name === name}"><div id="name" v-html="item.name" ></div><div id="thescore" v-html="item.score"></div></div></div>',
  mounted () {
    let self = this
    this.load()
    this.interval = setInterval(self.load, 3000)
  }
})

var app = new Vue({ 
  el: 'element',
  data: {
    generateUrl: '/pair',
    submitUrl: '/saveturing',
    loading: false,
    id: null,
    finished: false,
    score: null,
    name: null,
    uploading: false,
    interval: null,
    credits: false,
    censor: false,
    intro: true,
    storage: {
      log: [],
      questions: [],
    },
    scoreboard: [],
    done: false
  },
  computed: {
    lastQuestion () {
      if (!this.storage.questions) return false
      return this.storage.questions[this.storage.questions.length - 1]
    },
    answered () {
      if (this.storage.questions.length < 1) return -1
      return this.storage.questions.filter( function (x) { return x.answered }).length
    },
    totalQuestions () {
      return this.storage.questions.length > 10 ? this.storage.questions.length : 10
    }
  },
  methods: {
    start () {
      this.storeInBrowser();
    },
    clear () {
      this.finished = false
      this.storage = {log: [], questions: []}
      this.storeInBrowser()
    },
    generate () {
      document.activeElement.blur()
      if (!this.loading) {
        if (!this.storage.questions || !this.lastQuestion || this.lastQuestion.answered) {
          let self = this
          this.options = null
          this.selected = 0
          this.loading = true
          // define log object
          self.log('generate')
          let postdata = null
          if (self.lastQuestion) {
            postdata = {}
            postdata.iteration = self.lastQuestion.raw.iteration
            postdata.level = self.lastQuestion.raw.level
            postdata.seen = self.storage.questions.map( function (x) { return x.raw.id })
          }
          axios.post(self.generateUrl, postdata).then(function(res) {
            self.loading = false
            // setup new question
            let newq = {}
            // random type
            newq.artist = res.data.artist
            newq.album = res.data.album
            if (Math.random() > 0.5) {
              newq.type = 'choose'
              let shuffle = Math.round(Math.random()) + 1
              newq.line1 = shuffle === 1 ? res.data.real : res.data.fake
              newq.line2 = shuffle === 2 ? res.data.real : res.data.fake
              newq.answer = shuffle
              newq.raw = res.data
            } else {
              newq.type = 'forreal'
              newq.answer = Math.round(Math.random()) + 1 // 1 = real or 2 = fake
              newq.line = newq.answer === 1 ? res.data.real : res.data.fake
              newq.raw = res.data
            }
            // newquestion.received = new Date().getTime()
            if (!self.storage.questions) self.storage.questions = []
            newq.questiontime = 15000
            self.storage.questions.push(newq)
            self.log('new question received')
            self.storeInBrowser()
            self.starttimer(15000)
          }).catch(function (res) {
            self.loading = false
            self.log('error', {message: 'generate request failed', error: res })
            self.storeInBrowser()
          });
        } else {
          console.warn(`last question not yet answered...`)  
        }
      } else {
        console.warn(`don't try to generate again while loading...`)
      }
    },
    select (val) {
      if (this.storage.questions.length > 0 && this.lastQuestion && !this.lastQuestion.answered) {
        this.log('select', val)
        this.storage.questions[this.storage.questions.length - 1].selected = val
        this.storeInBrowser()
        this.$forceUpdate()
      }
    },
    starttimer (length) {
      var self = this;
      var progressbar = document.getElementById("progressbar");
      var progress = 0;
      var intervaltime = 10;
      progressbar.classList.remove('fifty')
      progressbar.classList.remove('seventyfive')
      progressbar.classList.remove('ninety')
      progressbar.classList.remove('done')
      if (self.interval) clearInterval(self.interval);
      self.interval = setInterval(function(){
        progress = progress + intervaltime
        if (progress >= length) clearInterval(self.interval);
        var perc = (progress/length) * 100
        if (perc > 50) progressbar.classList.add('fifty')
        if (perc > 75) progressbar.classList.add('seventyfive')
        if (perc > 90) progressbar.classList.add('ninety')
        if (perc == 100) {
          progressbar.classList.add('done')
          self.lastQuestion.correct = false
          self.lastQuestion.answered = true
          self.lastQuestion.answer = 'timeup'
          self.storeInBrowser()
          self.$forceUpdate()
          setTimeout(function () {
            if (!self.checkFinished()) {
              self.generate()
            } else {
              self.submit()
            }
          }, 1000)
        }
        progressbar.style.width = perc + '%'
      }, intervaltime)
    },
    submit () {
      let self = this
      self.log('submit') // for timestamp of submit
      // send to the server
      let upload = {}
      upload.log = {log: self.storage.log, questions: self.storage.questions}
      upload.score = self.storage.questions.filter(function(x) { return x.correct }).length /// calculate
      self.score = upload.score
      // console.log('uploading:',upload)
      self.uploading = true
      axios.post(self.submitUrl, upload).then(function(res) {
        self.name = res.data.name
        setTimeout(function(){
          self.clear()
          self.uploading = false
        }, 2000);
        // console.log('uploaded', {name: name}, res.data)
      }).catch(function (err) {
        self.log('error submitting', err)
        self.uploading = false
      })
    },
    check () {
      let self = this
      let last = this.storage.questions[this.storage.questions.length - 1]
      if (last.selected !== null && last.selected !== undefined) {
        self.log('answered')
        if (!last.answered || last.answered === undefined) {
          if (last.selected === last.answer) last.correct = true
          else last.correct = false
          last.answered = true
          this.storage.questions[this.storage.questions.length - 1] = last
          this.storeInBrowser()
          this.$forceUpdate()
          if (self.interval) clearInterval(self.interval)
          setTimeout(function () {
            if (!self.checkFinished()) {
              self.generate()
            } else {
              self.submit()
            }
          }, 3000)
        }
      } else {
        // console.log('no anwer given yet')
      }
    },
    checkFinished () {
      if (this.storage.questions.length <= 10) return false
      if (this.lastQuestion.correct) return false
      else {
        this.finished = true
        return true
      }
    },
    storeInBrowser () {
      localStorage.setItem('battle', JSON.stringify(this.storage));
    },
    log (type, message) {
      if (typeof message === "object") message = JSON.parse(JSON.stringify(message))
      let obj = {}
      obj.type = type
      obj.message = message || ''
      obj.timestamp = new Date().getTime();
      // console.log(obj.timestamp, type, obj.message)
      this.storage.log.push(obj)
    },
    dotclass (n) {
      n = n - 1
      if (!this.storage.questions[n]) return false
      if (this.storage.questions[n].correct) return 'correct'
      if (this.storage.questions[n].correct === false) return 'wrong'
      if (this.storage.questions[n] === this.lastQuestion) return 'active'
    },
    restart () {
      this.clear()
    }
  },
  created: function () {
    let self = this
    let ls = localStorage.getItem('battle')
    if (ls !== '' && ls !== null && ls !== undefined) { this.storage = JSON.parse(ls) }
    window.addEventListener('keydown', function(ev) {
      if (ev.keyCode === 37) self.select(1)
      if (ev.keyCode === 39) self.select(2)
      if (ev.keyCode === 32) self.start()
      if (ev.keyCode === 27) self.restart()
      if (ev.keyCode === 13) {
        if (self.storage.questions.length === 0) self.generate()
        else self.check()
      }
      /* magic keys */
      if (ev.keyCode === 74 && ev.shiftKey) { // J
        if (self.intro) {
          self.intro = false
        } else {
          self.select(1)
          if (self.storage.questions.length === 0) self.generate()
          else self.check()
        }
      }
      if (ev.keyCode === 75 && ev.shiftKey) { // K
        if (self.intro) {
          self.intro = false
        } else {
          self.select(2)
          if (self.storage.questions.length === 0) self.generate()
          else self.check()
        }
      }
      if (ev.keyCode === 76 && ev.shiftKey) { // L
        self.intro = false
        self.censor = false
      }
      if (ev.keyCode === 80 && ev.shiftKey) { // P
        self.intro = false
        self.censor = true
      }
    })
    // if(this.checkFinished()) this.clear()
    this.clear()
  }
});
