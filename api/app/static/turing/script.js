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
    storage: {
      log: [],
      questions: [],
    },
    scoreboard: [
      {name: 'Lil Turing', score: 17 },
      {name: 'MC FOGG', score: 15 },
      {name: 'Lil Hop', score: 13 },
      {name: 'MC Hip', score: 17 },
      {name: 'Lil ENrique', score: 17 },
      {name: 'MC Turing', score: 17 },
      {name: 'Lil MIC', score: 17 },
      {name: 'MC Mini', score: 17 },
      {name: 'Lil Miny', score: 17 },
      {name: 'MC Moe', score: 17 },
    ],
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
      this.score = 0
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
            console.log(res.data)
            self.loading = false
            // setup new question
            let newq = {}
            // random type
            if (Math.random() > 0.5) {
              newq.type = 'choose'
              let shuffle = Math.floor(Math.random() * 2) + 1
              newq.line1 = shuffle === 1 ? res.data.real : res.data.fake
              newq.line2 = shuffle === 2 ? res.data.real : res.data.fake
              newq.answer = shuffle
              newq.raw = res.data
            } else {
              newq.type = 'forreal'
              newq.answer = Math.floor(Math.random() * 2) + 1 // 1 = real or 2 = fake
              newq.line = newq.answer === 1 ? res.data.real : res.data.fake
              newq.raw = res.data
            }
            // newquestion.received = new Date().getTime()
            if (!self.storage.questions) self.storage.questions = []
            self.storage.questions.push(newq)
            self.log('new question received')
            self.storeInBrowser()
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
    submit () {
      let self = this
      self.log('submit') // for timestamp of submit
      // send to the server
      let upload = {}
      upload.log = {log: self.storage.log, questions: self.storage.questions}
      upload.score = self.storage.questions.filter(function(x) { return x.correct }).length /// calculate
      self.score = upload.score
      axios.post(self.submitUrl, upload).then(function(res) {
        /* 
        
        todo: set name here 
        
        */
        let name = ''
        var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (var i = 0; i < 5; i++) {
          name += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        self.name = name
        console.log('uploaded')
      }).catch(function (err) {
        self.log('error submitting', err)
        let name = ''
        var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        for (var i = 0; i < 5; i++) {
          name += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        self.name = name
        // setTimeout(submit, 1000)
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
          setTimeout(function () {
            if (!self.checkFinished()) {
              self.generate()
            } else {
              self.submit()
            }
          }, 1000)
        }
      } else {
        console.log('no anwer given yet')
      }
    },
    checkFinished () {
      if (this.storage.questions.length < 10) return false
      if (this.storage.questions.filter(x => x.correct).length === this.storage.questions.length && this.lastQuestion.answered) return false
      else {
        this.finished = true
        return true
      }
    },
    storeInBrowser () {
      localStorage.setItem('battle', JSON.stringify(this.storage));
    },
    log (type, message) {
      let obj = {}
      obj.type = type
      obj.message = message || ''
      obj.timestamp = new Date().getTime();
      console.log(obj.timestamp, obj.type, obj.message)
      this.storage.log.push(obj)
    },
    dotclass (n) {
      n = n - 1
      if (!this.storage.questions[n]) return false
      if (this.storage.questions[n].correct) return 'correct'
      if (this.storage.questions[n].correct === false) return 'wrong'
      if (this.storage.questions[n] === this.lastQuestion) return 'active'
    },
    getScoreboard () {
      axios.get('/scoreboard').then(function (res) {
        console.log('scoreboard working', res.data)
      }).catch(function(err) {
        console.log('error getting scoreboard', err)
      })
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
    })
    if(this.checkFinished()) this.clear()
  }
});