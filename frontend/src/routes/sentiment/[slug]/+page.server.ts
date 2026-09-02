import type { PageServerLoad } from './$types';

export const load : PageServerLoad = async ({ fetch, params }) => {
	const { slug } = params;
	const ticker = slug.toUpperCase();

	const res = await fetch(`http://127.0.0.1:8000/sentiment/${ticker}?limit=10`);
    const item = await res.json()

    console.log('Fetched sentiment item:', item);

	return { ticker, item };
}
