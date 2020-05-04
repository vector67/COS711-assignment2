num_buckets = 5
with open('../../output/best_order.csv', 'r') as f:
    frequency = [[] for x in range(1020)]
    k = 0
    for line in f:
        if k > 10:
            continue
        line_arr = line.split(',')
        # population.append([, float(line_arr[1])])
        genes = list(map(int, line_arr[0].split("|")))
        i = 0
        for gene in genes:
            frequency[gene].append(i)
            i += 1
        k += 1
    i = 0
    average_frequency = []
    for m in frequency:
        average_frequency.append((sum(m) / len(m), i))
        # print(frequency[i], sum(f) / len(f))
        i += 1
    average_frequency.sort(key=lambda x: x[0])
    print(average_frequency)
    total_matches = 0
    for i in range(len(average_frequency)):
        bucket = int(i / len(average_frequency) * num_buckets)
        matches = 0
        for freq in frequency[average_frequency[i][1]]:
            if bucket == int(freq / len(frequency) * num_buckets):
                matches += 1
        total_matches += matches
        print(i, frequency[average_frequency[i][1]], matches, average_frequency[i])
    print(frequency)
    print(total_matches / len(frequency))
# print(frequency)
