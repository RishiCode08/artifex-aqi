export default async function handler(req, res) {
  const { FIREBASE_DATABASE_URL, DEVICE_ID } = process.env;
  const { startKey, endKey } = req.query;

  let url = `${FIREBASE_DATABASE_URL}/devices/${DEVICE_ID}/readings.json?orderBy="$key"`;
  if (startKey) url += `&startAt="${startKey}"`;
  if (endKey)   url += `&endAt="${endKey}"`;

  try {
    const r = await fetch(url);
    if (!r.ok) return res.status(502).json({ error: 'Firebase error' });
    const data = await r.json();
    res.status(200).json(data || {});
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}