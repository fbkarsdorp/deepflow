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
        if (censored.indexOf(v.toLowerCase()) > -1) words[k] = '**********************************'.substr(0, v.length)
      })
      return words.join(' ')
    }
  },
  template: '<div>{{censored ? newtext : text}}</div>'
})

var app = new Vue({ 
  el: 'element',
  data: {
    statusUrl: '/status/',
    generateUrl: '/generate',
    submitUrl: '/upload',
    loading: false,
    selected: null,
    id: null,
    censored: false,
    storage: {
      lyric: [],
      log: [],
      generated: []
    },
    done: false
  },
  computed: {
    last () {
      return this.storage.generated[this.storage.generated.length -1]
    }
  },
  methods: {
    clear () {
      this.storage.lyric = []
      this.storage.log = []
      this.storage.generated = []
      this.id = null
      this.$forceUpdate()
      document.activeElement.blur()
      this.storeInBrowser()
      this.generate()
    },
    generate (resample) {
      if (resample === undefined) resample = false
      if (this.storage.lyric.length < 20) {
        if (!this.loading) {
          let self = this
          self.selected = null
          self.loading = true
          self.log('generate', resample)
          // send seed id
          let id = null
          if (this.storage.lyric.length > 0) {
            id = this.storage.lyric[this.storage.lyric.length - 1].id
          }
          axios.post(self.generateUrl,{seed_id: id, resample: resample}).then(function(res) {
            if (res.data.id) {
              self.id = res.data.id
              self.log('received job id', {jobid: res.data.id})
              self.storeInBrowser()
              setTimeout(self.polling, 1000)
            } else {
              self.loading = false
              self.log('error', 'did not receive job id, trying again in 1 sec')
              self.storeInBrowser()
              // setTimeout(self.generate, 1000)
            }
          }).catch(function (res) {
            self.loading = false
            self.log('error', {message: 'generate request failed', error: res })
            self.storeInBrowser()
            // retry?
          });
        } else {
          console.warn(`don't try to generate again while loading...`)
        }
      }
    },
    polling () {
      let self = this
      axios.get(self.statusUrl + self.id).then(function(res) {
        if (!res.data.status || res.data.status === 'busy') {
          // retry 
          setTimeout(self.polling, 500)
        } else {
          self.loading = false
          if (res.data.status === 'fail') {
            self.log('generate failed', res.data.message)
          }
          if (res.data.status === 'OK') {
            self.log('received results', {jobid: self.id})
            self.storage.generated.push(res.data.payload)
            self.storeInBrowser()
          }
        }
      }).catch(function (err) {
        self.log('error', 'http request failed /status/' + self.id)
        console.warn('could not fetch status of id:' + self.id, err)
        console.log('retry in 1 second')
        setTimeout(this.polling, 1000)
      });
    },
    add (item) {
      console.log(item)
      let storage = this.storage
      this.log('line added', {jobid: this.id, line: this.selected})
      item.timestamp = new Date().getTime()
      if (storage.lyric === undefined) storage.lyric = []
      storage.lyric.push(item)
      this.storage = storage
      this.$forceUpdate()
      this.generate()
    },
    submit () {
      let self = this
      self.log('submit') // for timestamp of submit
      // console.log(JSON.parse(JSON.stringify(self.storage)))
      // send to the server
      axios.post(self.submitUrl, self.storage).then(function() {
        self.clear()
      }).catch(function (err) {
        self.log('error submitting', err)
        setTimeout(submit, 1000)
      })
    },
    changeSelection (next) {
      if (this.storage.lyric.length < 20) {
        if (this.last && this.last.length > 0) {
          if (this.selected === null) this.selected = 0
          else this.selected = (this.selected + this.last.length - next) % this.last.length
          this.log('select', {nr: this.selected})
          this.storeInBrowser()
        } else {
          // this.log('error', 'tried to change selection without generated')
        }
      }
    },
    switchcensor () {
      this.censored = !this.censored
      if (this.censored) this.log('censor true', this.id)
      else this.log('censor false', this.id)
    },
    storeInBrowser () {
      localStorage.setItem('storage', JSON.stringify(this.storage));
    },
    log (type, message) {
      let obj = {}
      obj.type = type
      obj.message = message || ''
      obj.timestamp = new Date().getTime();
      console.log(obj.timestamp, obj.type, obj.message)
      this.storage.log.push(obj)
    }
  },
  created: function () {
    let self = this
    let ls = localStorage.getItem('storage')
    this.storage = (ls !== '' && ls !== null) ? JSON.parse(ls) : this.storage
    if (!this.storage.lyric || !this.storage.lyric.generated || this.storage.lyric.generated.length < 1) this.generate()
    setTimeout(this.show, 500)
    window.addEventListener('keydown', function(ev) {
      // if (ev.keyCode === 13) self.add(self.last[self.selected])
      // if (ev.keyCode === 32) self.generate(true)
      // if (ev.keyCode === 27) self.submit()
      
      if (ev.keyCode === 40 && self.storage.lyric.length < 10) self.changeSelection(-1)
      if (ev.keyCode === 38 && self.storage.lyric.length < 10) self.changeSelection(1)
      if (ev.keyCode === 107) self.generate(true)
      if (ev.keyCode === 13) {
        if (self.storage.lyric.length === 10 ) {
          self.submit()
        } else {
          self.add(self.last[self.selected])
        }
      } 
      if (ev.keyCode === 27) self.submit()
      if (ev.keyCode === 8) {
        self.censored = !self.censored
      }
    })
  }
});