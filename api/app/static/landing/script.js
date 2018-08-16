var app = new Vue({ 
  el: 'app',
  data () {
    return {
      credits: false,
      media: false
    }
  },
  created () {
    // fetch the data when the view is created and the data is
    // already being observed
    this.fetchData()
  },
  methods: {
    fetchData () {
      var self = this
      axios.get('/static/media.json').then(function(res){
        if (res.data) {
          self.media = res.data.media
        }
      })
    }
  }
});
