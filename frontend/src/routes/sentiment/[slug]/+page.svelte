<script>
    let {data} = $props();
</script>
<h1>
    {data.ticker}

    <ul>
        {#each data.item.results as sentiment}
            <li>{sentiment[0]} | {sentiment[1]}</li>
        {/each}
    </ul>
</h1>

