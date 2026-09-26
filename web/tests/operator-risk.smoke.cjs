// Run against a locally served built UI. Playwright is a QA tool, not an app dependency.
// PLAYWRIGHT_MODULE=/path/to/node_modules/playwright node web/tests/operator-risk.smoke.cjs
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

(async () => {
  const browser = await chromium.launch({ headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_PATH ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_PATH } : {}),
  });
  try {
    const base = process.env.FLOODVALUE_QA_URL || 'http://127.0.0.1:8017';
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const sessionResponse = await page.request.get(`${base}/api/session`);
    assert.equal(sessionResponse.status(), 200, 'serve a completed bundle');
    const original = await sessionResponse.json();
    const url = `${base}/#/analysis/${encodeURIComponent(original.metadata.chip_id)}`;
    await page.goto(url);
    await page.locator('.strategy-uncertainty').first().waitFor();
    assert.equal(await page.locator('.strategy-uncertainty').count(), 3);
    assert.equal(await page.locator('.risk-point').count(), 10);
    await page.locator('.risk-point').first().focus();
    await page.keyboard.press('Enter');
    assert.equal(await page.locator('.risk-point.selected').count(), 1);
    assert.equal(await page.locator('tbody tr.selected').count(), 1);
    assert.equal(await page.locator('.asset-detail button').textContent(), 'Показать на карте');
    const budget = page.locator('.analysis-budget input[type=number]');
    await budget.fill('0');
    await page.waitForFunction(() => document.querySelector('.analysis-metrics strong')?.textContent === '0');
    assert.match(await page.locator('.zone-list').textContent(), /зон в плане C нет/);

    // Synthetic API responses test missing estimates and the already exported D-035 policy.
    // No files, probabilities or model configuration on the server are modified.
    let allMissing = false;
    await page.route('**/api/session?*', async route => {
      const session = structuredClone(original);
      session.metadata.uncertainty = { method: 'entropy_p', units: 'биты, 0…1' };
      session.assets.features.forEach((asset, i) => {
        if (allMissing || i < 2) Object.assign(asset.properties, {
          status: i === 1 ? 'partial' : 'no_data', p_flood: '', expected_loss_rub: '', uncertainty: '', rank: '',
        });
      });
      session.metadata.expected_loss = { inspection_priority: allMissing ? null : {
        rule: 'QA: сохранённая очередность, без нового правила во фронтенде',
        order: [...session.assets.features].reverse().map((asset, i) => ({
          asset_id: asset.properties.asset_id, inspection_order: i + 1,
          group: asset.properties.status === 'ok' ? 'at_risk' : 'not_assessed',
          action: 'QA: проверить на месте', reason: 'QA: рекомендация из комплекта',
        })),
      } };
      await route.fulfill({ json: session });
    });
    await page.reload();
    await page.getByRole('button', { name: 'По очередности осмотра', exact: true }).waitFor();
    assert.equal(await page.locator('.risk-point').count(), 8);
    const missing = page.locator('.analysis-body > .risk-missing');
    assert.match(await missing.textContent(), /недостаточно данных · 2/);
    const firstId = original.assets.features.at(-1).properties.asset_id;
    assert.match(await page.locator('tbody tr').first().textContent(), new RegExp(firstId));
    await missing.locator('button').first().click();
    assert.match(await page.locator('.asset-detail').textContent(), /точное p нет оценки/);
    assert.equal((await page.locator('tbody tr.selected td').allTextContents()).slice(1).every(value => value.trim() === '—'), true);
    await page.locator('.strategy-uncertainty').first().waitFor();
    assert.match(await page.locator('.strategy-uncertainty').first().textContent(), /бит · сумма по объектам/);
    assert.doesNotMatch(await page.locator('.strategy-uncertainty').first().textContent(), /0…1/);
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.evaluate(() => document.fonts.ready);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert.ok(overflow <= 1, `page overflow at ${width}px: ${overflow}`);
      await page.locator('.risk-object-list button').first().click();
      assert.equal(await page.locator('.risk-point.selected').count(), 1);
    }
    allMissing = true;
    await page.reload();
    await page.getByText('Нет оценённых объектов для диаграммы.', { exact: true }).waitFor();
    assert.match(await page.locator('.analysis-columns > section > .card-title').textContent(), /0 из 10 оценено · ущерб —/);
    assert.equal(await page.locator('.risk-point').count(), 0);
    assert.equal(await page.locator('.monte-carlo').count(), 0);
    assert.deepEqual(errors, []);
    console.log('PASS: A/B/C, budget, chart/table selection, saved inspection order, partial/no_data, all-missing, 390/320px, no console errors');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
