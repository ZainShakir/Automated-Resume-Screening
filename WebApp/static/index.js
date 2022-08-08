function deleteJob(jobId) {
    fetch("/delete-job", {
      method: "POST",
      body: JSON.stringify({ jobId: jobId }),
    }).then((_res) => {
      window.location.href = "/";
    });
  }
function showJD(a){
    let features = 'menubar=yes,location=yes,resizable=no,scrollbars=yes,status=no,height=700,width=1000';
    let url = '/static/jd/'+a;

    window.open(url,'_blank',features);
}