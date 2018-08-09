censored = ["word","nigga","shit","fuck","bitch","niggas","niggaz","black","fuckin","bitches","hoes","dick","blow","pussy","fucking","motherfucker","fucked","motherfuckin","ho","motherfuckers","pimp","bullshit","cock","motherfucking","blowin","pimpin","booty","muthafuckin","bangin","muthafucka","blowing","fucker","blown","muthafuckas","cocked","cum","allah","blows","titties","suckin","fuckers","whore","faggot","mothafuckin","banging","jigga","whores","fucks","mothafucka","tits","gay","shits","slut","blacks","dicks","motherfuck","pimping","fucka","shitty","rape","faggots","coochie","african","mothafuckas","africa","virgin","doggy","shittin","fu","hooker","nigger","blowed","motherfucka","stripper","pum","terrorist","pussies","sluts","womb","afro","clit","racist","biatch","porn","penetrate","penis","hos","fuckas","homo","condom","fag","niggers","raped","bitchin","strippers","bullshittin","nig","porno","titty","vagina","sperm","cunt","dyke","muthafucking","banged","fags","german","gangbang","puss","shitted","holocaust","jews","motherfuckas","muhammad","cockin","fukin","jew","muthafucker","sexin","jerkin","mothafucker","prostitute","turk","cocks","dykes","bitchy","muthafuck","jewish","shitting","terrorists","muthafuckers","nipple","erection","germany","israel","gangbangin","mothafucking","nazi","fuk","mothafuckers","cocking","bitching","muffin","prostitutes","mutherfuckin","africans","beyotch","cummin","fucken","niggy","nympho","penetration","anal","cocksucker","dickin","lesbian","raping","dickie","muthafuckaz","rapin","clitoris","hoez","dicky","fucc","gangbangers","cunts","hoein","rapist","bitchez","terrorism","booby","mutherfucker","prostitution","tittie","afrika","culo","boobs","mutherfuckers","blackman","bullshitting","clits","cocksuckers","sexing","fuckaz","shiiit","mothafuck","vaginas","arse","penetentiary","mothafuckaz","cumin","mutherfucking","jizz","sexed","dicked","blowjob","muhfuckin","nazis","gangbanger","scud","shiit","slutty","bitched","homos","sexist","mohammed","penetrated","israeli","lesbians","penetrating","boner","dickhead","biiitch","boobies","cumming","shitter","sudan","tity","mothafuckn","muhammed","niggah","fuckery","nymphos","beatch","bitchs","fuccin","germans","vaginal","muhfucka","homosexual","dickory","fuckn","jiggaboo","rapists","hoeing","muhfucker","pussyhole","hookie","gangbanging","muffins","pedophile","pornographic","boob","fucky","sexes","shitlist","butthole","pedo","pornos","rapes","jiggable","bisexual","muff","stompdashitoutu","tities","bitchass","boobie","brothel","israelite","mufucka","dickey","dicking","pussys","sexo","shiiiit","mufuckin","pecker","erections","fucca","fuga","homosexuals","israelites","motherfuckaz","shite","skanks","beyatch","mufuckas","niggaro","peckerwood","bullshitin","bullshitters","pornography","shithead","blowjobs","bullshitted","motherfucken","pornstar","jiggas","muthafucken","cums","muhfuckas","nevehoe","niggaaa","niggs","transvestite"];

Vue.component('flow-line', {
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
          self.log('generate')
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
      if (ev.keyCode === 40) self.changeSelection(-1)
      if (ev.keyCode === 38) self.changeSelection(1)
      if (ev.keyCode === 13) self.add(self.last[self.selected])
      if (ev.keyCode === 32) self.generate(true)
      if (ev.keyCode === 27) self.submit()
    })
  }
});