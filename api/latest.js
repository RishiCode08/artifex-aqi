export default async function handler(req, res) {
  const { FIREBASE_DATABASE_URL, DEVICE_ID } = process.env;
  const url = `${FIREBASE_DATABASE_URL}/devices/${DEVICE_ID}/readings.json?orderBy="$key"&limitToLast=1`;

  try {
    const r = await fetch(url);
    if (!r.ok) return res.status(502).json({ error: 'Firebase error' });
    const data = await r.json();
    res.status(200).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}