(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon = (name, extra = '') => `<svg class="icon ${extra}" aria-hidden="true"><use href="#icon-${name}"/></svg>`;
  const videoPath = id => `assets/videos/demo-${id}.mp4`;
  const posterPath = id => `assets/images/posters/demo-${id}.jpg`;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const dialog = $('#video-dialog');
  const dialogVideo = $('#dialog-video');
  let galleryPaused = reducedMotion.matches;
  let heroPaused = reducedMotion.matches;
  let heroVisible = true;
  let dialogOpener = null;

  const realDemos = [
    {id:19, title:'Pick A4 paper', category:'dexterous', featured:true, success:93.3, description:'Friction, grasping, and precise placement.'},
    {id:21, title:'Shuttlecock insertion', category:'dexterous', featured:true, success:60.0, description:'Delicate grasping and controlled contact.'},
    {id:20, title:'Express packing', category:'bimanual', featured:true, success:73.3, description:'Coordinated manipulation across task stages.'},
    {id:22, title:'Powered screw driving', category:'bimanual', featured:true, success:60.0, description:'Contact-sensitive, bimanual tool use.'},
    {id:23, title:'Paper manipulation · rollout 2', category:'dexterous', description:'Additional experiment recording.'},
    {id:24, title:'Paper manipulation · rollout 3', category:'dexterous', description:'Additional experiment recording.'},
    {id:25, title:'Paper manipulation · side view', category:'dexterous', description:'Additional experiment recording.'},
    {id:26, title:'Screw driving · rollout 2', category:'bimanual', description:'Additional experiment recording.'},
    {id:27, title:'Shuttlecock manipulation · rollout 2', category:'dexterous', description:'Additional experiment recording.'},
    {id:28, title:'Screw driving · rollout 3', category:'bimanual', description:'Additional experiment recording.'},
    {id:29, title:'Screw driving · rollout 4', category:'bimanual', description:'Additional experiment recording.'},
    {id:30, title:'Paper manipulation · rollout 4', category:'dexterous', description:'Additional experiment recording.'},
  ];
  // Source videos are numbered only; simulation captions deliberately avoid
  // claiming an unverified correspondence between file number and task name.
  const simDemos = Array.from({length:18}, (_, index) => ({
    id:index + 1,
    title:index < 8 ? `Contact-rich task ${String(index + 1).padStart(2,'0')}` : `Long-horizon task ${String(index - 7).padStart(2,'0')}`,
    category:index < 8 ? 'univtac' : 'libero',
    description:index < 8 ? 'UniVTAC · vision + touch' : 'LIBERO-Long · vision only',
  }));
  const allDemos = [...realDemos, ...simDemos];
  const categories = {dexterous:'Dexterous hands', bimanual:'Bimanual grippers', univtac:'UniVTAC', libero:'LIBERO-Long'};

  function setPressed(buttons, active) {
    buttons.forEach(button => {
      const selected = button === active;
      button.classList.toggle('is-active', selected);
      button.setAttribute('aria-pressed', String(selected));
    });
  }

  function play(video) {
    if (video.readyState === 0 && !video.getAttribute('src') && !video.querySelector('source')) return;
    const result = video.play();
    if (result) result.catch(() => {}); // Native controls remain available if autoplay is blocked.
  }

  function mayPlay(video) {
    return !galleryPaused && !document.hidden && !dialog.open && video.dataset.visible === 'true' && video.dataset.userPaused !== 'true';
  }

  const mediaObserver = new IntersectionObserver(entries => {
    for (const {target:video, isIntersecting} of entries) {
      video.dataset.visible = String(isIntersecting);
      if (isIntersecting && video.dataset.src) {
        video.src = video.dataset.src;
        delete video.dataset.src;
        video.load();
      }
      if (isIntersecting && mayPlay(video)) play(video);
      else video.pause();
    }
  }, {threshold:0.15});

  function observeMedia(root) {
    $$('video[data-managed]', root).forEach(video => {
      video.muted = true;
      video.addEventListener('pause', () => {
        if (video.dataset.visible === 'true' && !galleryPaused && !document.hidden && !dialog.open && video.isConnected) {
          video.dataset.userPaused = 'true';
        }
      });
      video.addEventListener('play', () => { delete video.dataset.userPaused; });
      mediaObserver.observe(video);
    });
  }

  function clearMedia(root) {
    $$('video', root).forEach(video => {
      video.dataset.visible = 'false';
      mediaObserver.unobserve(video);
      video.pause();
      video.removeAttribute('src');
      video.load();
    });
  }

  function renderDemos(filter = 'featured') {
    const container = $('#demo-grid');
    clearMedia(container);
    const demos = realDemos.filter(demo => filter === 'all' || (filter === 'featured' ? demo.featured : demo.category === filter));
    container.innerHTML = demos.map(demo => `
      <article class="demo-card">
        <div class="demo-media">
          <video data-managed data-src="${videoPath(demo.id)}" poster="${posterPath(demo.id)}" muted loop playsinline controls preload="none" aria-label="${escape(demo.title)} experiment recording"></video>
          <span class="clip-tag">${categories[demo.category].toUpperCase()}</span>
          <button class="icon-button expand-video" data-open-video="${demo.id}" aria-label="Expand ${escape(demo.title)} video">${icon('expand')}</button>
        </div>
        <div class="demo-caption"><div><h3>${escape(demo.title)}</h3><p>${demo.description}</p></div>
          ${demo.success !== undefined ? `<div class="demo-success"><span>${demo.success.toFixed(1)}%</span><small>TASK SUCCESS</small></div>` : ''}
        </div>
      </article>`).join('');
    observeMedia(container);
  }

  $$('[data-demo-filter]').forEach(button => button.addEventListener('click', () => {
    setPressed($$('[data-demo-filter]'), button);
    renderDemos(button.dataset.demoFilter);
  }));

  function updateMotionButton() {
    $('.gallery-motion').innerHTML = `${icon(galleryPaused ? 'play' : 'pause', 'icon-small')}<span>${galleryPaused ? 'Play videos' : 'Pause videos'}</span>`;
    $('.gallery-motion').setAttribute('aria-label', galleryPaused ? 'Play visible experiment videos' : 'Pause all experiment videos');
  }
  $('.gallery-motion').addEventListener('click', () => {
    galleryPaused = !galleryPaused;
    updateMotionButton();
    $$('video[data-managed]').forEach(video => {
      if (galleryPaused) video.pause();
      else { delete video.dataset.userPaused; if (mayPlay(video)) play(video); }
    });
  });

  function renderSimulation(category = 'univtac') {
    const track = $('#sim-track');
    clearMedia(track);
    track.innerHTML = simDemos.filter(demo => demo.category === category).map((demo, index) => `
      <article class="sim-card"><div class="sim-media">
        <video data-managed data-src="${videoPath(demo.id)}" poster="${posterPath(demo.id)}" muted loop playsinline preload="none" aria-label="${demo.title} recording"></video>
        <span class="sim-index">${String(index + 1).padStart(2,'0')}</span>
        <button data-open-video="${demo.id}" aria-label="Play ${categories[category]} ${demo.title.toLowerCase()}">${icon('play')}</button>
      </div><h4>${demo.title}</h4><p>${demo.description.toUpperCase()}</p></article>`).join('');
    track.scrollLeft = 0;
    observeMedia(track);
    requestAnimationFrame(updateCarouselArrows);
  }
  $$('[data-sim]').forEach(button => button.addEventListener('click', () => {
    setPressed($$('[data-sim]'), button);
    renderSimulation(button.dataset.sim);
  }));
  function updateCarouselArrows() {
    const track = $('#sim-track');
    $('#sim-prev').disabled = track.scrollLeft <= 2;
    $('#sim-next').disabled = track.scrollLeft >= track.scrollWidth - track.clientWidth - 2;
  }
  $('#sim-track').addEventListener('scroll', updateCarouselArrows, {passive:true});
  function scrollSimulation(direction) {
    const track = $('#sim-track');
    const card = $('.sim-card', track);
    const distance = card.getBoundingClientRect().width + parseFloat(getComputedStyle(track).gap);
    track.scrollBy({left:direction * distance, behavior:reducedMotion.matches ? 'instant' : 'smooth'});
  }
  $('#sim-prev').addEventListener('click', () => scrollSimulation(-1));
  $('#sim-next').addEventListener('click', () => scrollSimulation(1));

  // Full-size viewing uses a native modal dialog, including Escape and focus return.
  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-open-video]');
    if (!trigger) return;
    const demo = allDemos.find(item => item.id === Number(trigger.dataset.openVideo));
    if (!demo) return;
    dialogOpener = trigger;
    $('#dialog-title').textContent = demo.title;
    $('#dialog-category').textContent = categories[demo.category].toUpperCase();
    $('#dialog-caption').textContent = demo.description;
    dialogVideo.src = videoPath(demo.id);
    dialogVideo.poster = posterPath(demo.id);
    dialog.showModal();
    document.body.classList.add('modal-open');
    $$('video[data-managed]').forEach(video => video.pause());
    $('#hero-video').pause();
    play(dialogVideo);
  });
  $('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    dialogVideo.pause();
    dialogVideo.removeAttribute('src');
    dialogVideo.load();
    document.body.classList.remove('modal-open');
    if (dialogOpener?.isConnected) dialogOpener.focus({preventScroll:true});
    $$('video[data-managed]').forEach(video => { if (mayPlay(video)) play(video); });
    if (!heroPaused && heroVisible && !document.hidden) play($('#hero-video'));
  });

  // Interactive vocabulary is an explanatory illustration, not recorded model output.
  const vocabulary = [
    ['approach','Move toward the object or region of interaction.'],
    ['move','Transport an end-effector or a held object.'],
    ['pick','Establish a grasp and lift an object.'],
    ['place','Set a held object onto a target surface or into a receptacle.'],
    ['flip','Turn an object over.'],
    ['rotate','Change an object’s orientation.'],
    ['toggle','Change the state of a switch or mechanism.'],
    ['open','Open an articulated object or container.'],
    ['close','Close an articulated object or container.'],
    ['push','Apply force to move an object away.'],
    ['pull','Apply force to draw an object toward the robot.'],
    ['insert','Guide an object into an opening or receptacle.'],
    ['press','Apply sustained contact force to a target.'],
    ['click','Activate a target with a brief press.'],
    ['strike','Make a brief impact with a target.'],
    ['handover','Transfer an object between end-effectors or agents.'],
  ];
  $('.word-selector').innerHTML = vocabulary.map(([word]) => `<button data-word="${word}" class="${word === 'pick' ? 'is-active' : ''}" aria-pressed="${word === 'pick'}">${word}</button>`).join('');
  $$('[data-word]').forEach(button => button.addEventListener('click', () => {
    setPressed($$('[data-word]'), button);
    const word = button.dataset.word;
    $('.word-definition-name').textContent = word;
    $('.word-definition-text').textContent = vocabulary.find(item => item[0] === word)[1];
    $$('.selected-word').forEach(label => { label.textContent = word; });
  }));

  // All reported numbers below are transcribed from the supplied manuscript.
  const univtacRows = [
    ['ACTTac',71,56,24,99,46,31,28,29,48.0],
    ['VITaL',72,34,25,100,47,32,6,8,40.5],
    ['π0.5',100,74,25,49,35,34,8,72,49.6],
    ['StarVLA-α',32,69,52,68,51,88,24,65,56.1],
    ['InternVLA-A1',58,95,93,86,66,37,12,90,67.1],
    ['FTP-1',97,79,64,99,48,47,4,65,62.9],
    ['N0-VTLA',99,99,95,100,99,60,25,88,83.1],
    ['T-Rex',99,95,100,95,99,79,8,75,81.3],
    ['PhysWord',100,100,100,100,100,93,34,95,90.3],
  ];
  const liberoRows = [['OpenVLA',53.7],['π0.5',92.4],['OpenVLA-OFT',94.5],['GR00T-N1.7',94.4],['LaST0',95.6],['Motus',97.6],['Fast-WAM',95.2],['InternVLA-A1',98.4],['UniWAM',98.4],['PhysWord',99.6]];
  const realRows = [['π0.5',53.3,43.3,46.7,40.0,45.8],['FTP-1',66.7,56.7,50.0,46.7,55.0],['PhysWord',93.3,60.0,73.3,60.0,71.7]];
  const unseenRows = [['π0.5',46.7,40.0,13.3,26.7,20.0,26.7,28.9,'−38%'],['FTP-1',60.0,46.7,26.7,23.3,23.3,30.0,35.0,'−38%'],['PhysWord',90.0,83.3,80.0,43.3,36.7,46.7,63.3,'−17%']];
  function table(caption, headings, rows) {
    return `<table><caption>${caption}</caption><thead><tr>${headings.map(label => `<th scope="col">${label}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr class="${row[0] === 'PhysWord' ? 'ours' : ''}">${row.map((value,index) => index === 0 ? `<th scope="row">${escape(value)}</th>` : `<td>${typeof value === 'number' ? value.toFixed(1) : value}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  }
  const benchmarkViews = {
    simulation:{
      charts:[
        {title:'UniVTAC',subtitle:'8 tasks · vision + touch',tag:'VISUOTACTILE',rows:[['π0.5',49.6],['FTP-1',62.9],['T-Rex',81.3],['N0-VTLA',83.1],['PhysWord',90.3]],footer:'<strong>+7.2 percentage points</strong> over N0-VTLA.'},
        {title:'LIBERO-Long',subtitle:'10 tasks · vision only',tag:'LONG-HORIZON',rows:[['π0.5',92.4],['LaST0',95.6],['InternVLA-A1',98.4],['UniWAM',98.4],['PhysWord',99.6]],footer:'<strong>+1.2 percentage points</strong> over InternVLA-A1 and UniWAM (98.4%).'},
      ],
      note:'Selected baseline comparisons from Table 1. PhysWord is evaluated over three runs of 100 rollouts per task, using 50 training demonstrations per task. Bars start at zero; the complete numerical comparisons are available below.',
      table:() => table('UniVTAC · success rate (%) · Table 1',['Method','Lift bottle','Insert tube','Insert hole','Grasp / classify','Pull out key','Put bottle','Insert HDMI','Lift can','Mean'],univtacRows) + table('LIBERO-Long · success rate (%) · Table 1',['Method','Success rate'],liberoRows),
    },
    real:{
      charts:[
        {title:'Pick A4 paper',subtitle:'Slide, grasp, and place the whole sheet',tag:'TASK 1',rows:[['π0.5',53.3],['FTP-1',66.7],['PhysWord',93.3]],footer:'<strong>93.3%</strong> versus 53.3% (π0.5) and 66.7% (FTP-1).'},
        {title:'Shuttlecock insertion',subtitle:'Grasp, insert, and palm press',tag:'TASK 2',rows:[['π0.5',43.3],['FTP-1',56.7],['PhysWord',60.0]],footer:'<strong>60.0%</strong> versus 43.3% (π0.5) and 56.7% (FTP-1).'},
        {title:'Express packing',subtitle:'Pack, close the flaps, and seal',tag:'TASK 3',rows:[['π0.5',46.7],['FTP-1',50.0],['PhysWord',73.3]],footer:'<strong>73.3%</strong> versus 46.7% (π0.5) and 50.0% (FTP-1).'},
        {title:'Powered screw driving',subtitle:'Place the screw, then drive it in',tag:'TASK 4',rows:[['π0.5',40.0],['FTP-1',46.7],['PhysWord',60.0]],footer:'<strong>60.0%</strong> versus 40.0% (π0.5) and 46.7% (FTP-1).'},
      ],
      note:'Table 2(a). Each chart is one task. Each task uses 100 training demonstrations and three runs of 30 trials with matched initial configurations. PhysWord’s average per-task standard deviation is 2.5 percentage points.',
      table:() => table('Real-world manipulation · success rate (%) · Table 2(a)',['Method','Pick paper','Shuttlecock insert','Express packing','Screw driving','Mean'],realRows),
    },
    unseen:{
      charts:[
        {title:'Pick A4 paper',subtitle:'Three generalization settings',tag:'PAPER',groups:[
          {name:'Background',rows:[['π0.5',46.7],['FTP-1',60.0],['PhysWord',90.0]]},
          {name:'Object',rows:[['π0.5',40.0],['FTP-1',46.7],['PhysWord',83.3]]},
          {name:'Position',rows:[['π0.5',13.3],['FTP-1',26.7],['PhysWord',80.0]]},
        ],footer:'PhysWord holds <strong>90.0% / 83.3% / 80.0%</strong> on background, object, and position shifts.'},
        {title:'Powered screw driving',subtitle:'Three generalization settings',tag:'SCREW',groups:[
          {name:'Background',rows:[['π0.5',26.7],['FTP-1',23.3],['PhysWord',43.3]]},
          {name:'Object',rows:[['π0.5',20.0],['FTP-1',23.3],['PhysWord',36.7]]},
          {name:'Position',rows:[['π0.5',26.7],['FTP-1',30.0],['PhysWord',46.7]]},
        ],footer:'PhysWord holds <strong>43.3% / 36.7% / 46.7%</strong> on background, object, and position shifts.'},
      ],
      note:'Table 2(b). Each chart is one task. Background, object, and position are varied separately, and every method sees the same test configurations.',
      table:() => table('Unseen configurations · success rate (%) · Table 2(b)',['Method','Paper / background','Paper / object','Paper / position','Screw / background','Screw / object','Screw / position','Mean','Drop'],unseenRows),
    },
  };
  function barRows(rows) {
    return rows.map(([name,value],index) => `<div class="bar-row ${name === 'PhysWord' ? 'is-ours' : ''}" aria-hidden="true"><span class="bar-name">${escape(name)}</span><div class="bar-track"><div class="bar-fill" style="width:${value}%;animation-delay:${index * 40}ms"></div></div><span class="bar-value">${value.toFixed(1)}%</span></div>`).join('');
  }
  function chartSummary(chart) {
    const groups = chart.groups || [{name:'',rows:chart.rows}];
    return groups.map(group => `${group.name ? group.name + ': ' : ''}${group.rows.map(([name,value]) => `${name} ${value.toFixed(1)} percent`).join(', ')}`).join('; ');
  }
  function chartBars(chart) {
    if (!chart.groups) return barRows(chart.rows);
    return chart.groups.map(group => `<div class="scenario"><span class="scenario-name">${escape(group.name)}</span>${barRows(group.rows)}</div>`).join('');
  }
  function renderBenchmark(key = 'simulation') {
    const view = benchmarkViews[key];
    $('#benchmark-charts').innerHTML = view.charts.map(chart => `
      <article class="chart-card"><div class="chart-top"><div><h3>${chart.title}</h3><p>${chart.subtitle}</p></div><span class="chart-tag">${chart.tag}</span></div>
        <div class="bar-chart" role="img" aria-label="${escape(chart.title)} success rates: ${escape(chartSummary(chart))}">
          ${chartBars(chart)}
          <div class="bar-axis" aria-hidden="true"><span>0</span><span>50</span><span>100%</span></div>
        </div><p class="chart-bottom">${chart.footer}</p></article>`).join('');
    $('#benchmark-note').textContent = view.note;
    $('#benchmark-table').innerHTML = view.table();
  }
  $$('[data-benchmark]').forEach(button => button.addEventListener('click', () => {
    setPressed($$('[data-benchmark]'), button);
    renderBenchmark(button.dataset.benchmark);
  }));

  // Hero scene selection and respectful autoplay.
  const heroVideo = $('#hero-video');
  // Playback is controlled below so changing scenes preserves a manual pause.
  heroVideo.autoplay = false;
  function updateHeroMotion() {
    const button = $('.hero-motion');
    button.innerHTML = icon(heroPaused ? 'play' : 'pause');
    button.setAttribute('aria-label', heroPaused ? 'Play background video' : 'Pause background video');
  }
  if (heroPaused) { heroVideo.removeAttribute('autoplay'); heroVideo.pause(); }
  $('.hero-motion').addEventListener('click', () => {
    heroPaused = !heroPaused;
    updateHeroMotion();
    if (heroPaused) heroVideo.pause(); else play(heroVideo);
  });
  $$('.hero-scene').forEach(button => button.addEventListener('click', () => {
    if (button.classList.contains('is-active')) return;
    setPressed($$('.hero-scene'), button);
    heroVideo.classList.add('is-switching');
    heroVideo.poster = posterPath(button.dataset.video);
    heroVideo.src = videoPath(button.dataset.video);
    heroVideo.load();
    heroVideo.addEventListener('loadeddata', () => {
      heroVideo.classList.remove('is-switching');
      if (!heroPaused && !document.hidden) play(heroVideo);
    }, {once:true});
  }));
  new IntersectionObserver(entries => {
    heroVisible = entries[0].isIntersecting;
    if (heroVisible && !heroPaused && !document.hidden && !dialog.open) play(heroVideo);
    else heroVideo.pause();
  }, {threshold:0.1}).observe($('#home'));
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      $$('video').forEach(video => video.pause());
    } else {
      if (dialog.open) play(dialogVideo);
      else {
        if (!heroPaused && heroVisible) play(heroVideo);
        $$('video[data-managed]').forEach(video => { if (mayPlay(video)) play(video); });
      }
    }
  });

  const nav = $('.site-nav');
  const menuButton = $('.menu-toggle');
  function closeMenu() {
    nav.classList.remove('menu-open');
    menuButton.setAttribute('aria-expanded','false');
    menuButton.setAttribute('aria-label','Open navigation');
    menuButton.innerHTML = icon('menu');
  }
  menuButton.addEventListener('click', () => {
    const open = nav.classList.toggle('menu-open');
    menuButton.setAttribute('aria-expanded',String(open));
    menuButton.setAttribute('aria-label',open ? 'Close navigation' : 'Open navigation');
    menuButton.innerHTML = icon(open ? 'close' : 'menu');
  });
  $$('.nav-links a').forEach(link => link.addEventListener('click',closeMenu));
  document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
  document.addEventListener('click', event => { if (!event.composedPath().includes(nav)) closeMenu(); });
  const navSections = $$('.nav-links a').map(link => ({link,section:$(link.getAttribute('href'))}));
  let scrollScheduled = false;
  function updateNavigation() {
    nav.classList.toggle('is-solid', window.scrollY > $('#home').offsetHeight - nav.offsetHeight);
    let current;
    for (const item of navSections) if (item.section.getBoundingClientRect().top <= 170) current = item;
    navSections.forEach(item => { if (item === current) item.link.setAttribute('aria-current','location'); else item.link.removeAttribute('aria-current'); });
    scrollScheduled = false;
  }
  window.addEventListener('scroll', () => { if (!scrollScheduled) { scrollScheduled = true; requestAnimationFrame(updateNavigation); } }, {passive:true});
  window.addEventListener('resize', () => { updateNavigation(); updateCarouselArrows(); if (window.innerWidth > 640) closeMenu(); }, {passive:true});

  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(({target,isIntersecting}) => {
      if (isIntersecting) { target.classList.add('is-visible'); revealObserver.unobserve(target); }
    });
  }, {threshold:0.08});
  $$('[data-reveal]').forEach(element => revealObserver.observe(element));

  let citationTimer;
  $('#copy-citation').addEventListener('click', async () => {
    const content = $('#bibtex').textContent;
    let copied = false;
    try { await navigator.clipboard.writeText(content); copied = true; }
    catch {
      const input = document.createElement('textarea');
      input.value = content;
      input.setAttribute('readonly','');
      input.style.cssText = 'position:fixed;left:0;top:0;opacity:0;pointer-events:none';
      document.body.appendChild(input);
      input.select();
      copied = document.execCommand('copy');
      input.remove();
    }
    const button = $('#copy-citation');
    if (copied) {
      button.innerHTML = `${icon('check','icon-small')}<span>Copied!</span>`;
      $('#copy-status').textContent = 'Citation copied to clipboard.';
    } else {
      const range = document.createRange();
      range.selectNodeContents($('#bibtex'));
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.innerHTML = `${icon('copy','icon-small')}<span>Selected — press copy</span>`;
      $('#copy-status').textContent = 'Citation selected. Use your keyboard copy command.';
    }
    clearTimeout(citationTimer);
    citationTimer = setTimeout(() => { button.innerHTML = `${icon('copy','icon-small')}<span>Copy citation</span>`; $('#copy-status').textContent = ''; }, 2500);
  });

  renderDemos();
  renderSimulation();
  renderBenchmark();
  updateMotionButton();
  updateHeroMotion();
  updateNavigation();
})();
